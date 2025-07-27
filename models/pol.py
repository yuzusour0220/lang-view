import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video_transformer_egovlp import EgoVLPv2
from common.utils import *


class videoEncoder(nn.Module):
	def __init__(self, kwargs):
		super().__init__()

		self.recog_arc = kwargs["recog_arc"]
		vidEncoder_ckptPath = kwargs["vidEncoder_ckptPath"]
		self.num_frames = kwargs["num_frames"]
		self.use_relativeCameraPoseLoss = kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False
		self.relativeCameraPoseLoss_refType = kwargs["relativeCameraPoseLoss_refType"] if ("relativeCameraPoseLoss_refType" in kwargs) else "first_view"

		if vidEncoder_ckptPath is not None:
			assert ospif(vidEncoder_ckptPath), print(vidEncoder_ckptPath, "does not exist")

		assert self.recog_arc in ["egovlp_v2", ]
		self.vid_encoder = None
		if self.recog_arc == "egovlp_v2":
			self.vid_encoder = EgoVLPv2(ckpt_path=vidEncoder_ckptPath, num_frames=self.num_frames,
										kwargs=kwargs)
		assert self.vid_encoder

	def forward(self, frms, ):
		B, nm_vws, t = frms.shape[0], frms.shape[1], frms.shape[2]
		frms = frms.permute((0, 1, 2, 5, 3, 4))
		frms = frms.reshape((frms.shape[0] * frms.shape[1], 
							 frms.shape[2],
							 frms.shape[3],
							 frms.shape[4],
							 frms.shape[5]))

		if self.use_relativeCameraPoseLoss:
			fts, fts_relCameraPose = self.vid_encoder(frms)
			
		else:
			fts = self.vid_encoder(frms)

		assert len(fts.shape) == 2		
		fts = fts.reshape((B , nm_vws, fts.shape[1]))
		fts = fts.reshape((B, fts.shape[1] * fts.shape[2]))
		if self.use_relativeCameraPoseLoss:
			assert len(fts_relCameraPose.shape) == 3	
			if self.relativeCameraPoseLoss_refType == "all_views":	
				fts_relCameraPose = fts_relCameraPose.reshape((B , nm_vws ** 2, -1, fts_relCameraPose.shape[2]))
			else:
				fts_relCameraPose = fts_relCameraPose.reshape((B , nm_vws, -1, fts_relCameraPose.shape[2]))

			return fts, fts_relCameraPose
		else:
			return fts


class pol_v1(nn.Module):
	def __init__(self, kwargs):
		super().__init__()

		self.recog_arc = kwargs["recog_arc"]
		self.use_transformerPol = kwargs["use_transformerPol"] if ("use_transformerPol" in kwargs) else False
		self.numLayers_transformerPol = kwargs["numLayers_transformerPol"] if ("numLayers_transformerPol" in kwargs) else 2
		self.transformerPol_dropout = kwargs["transformerPol_dropout"] if ("transformerPol_dropout" in kwargs) else 0.
		self.addPE_transformerPol = kwargs["addPE_transformerPol"] if ("addPE_transformerPol" in kwargs) else False
		self.linearLayer_dims = kwargs["linearLayer_dims"]
		self.linearLayer_dropout = kwargs["linearLayer_dropout"]
		self.task_type = kwargs["task_type"]
		self.num_frames = kwargs["num_frames"]
		self.all_views = kwargs["all_views"]
		self.use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
																False
		self.use_egoVlpV2_takeVideoFeats_usingCenterTime = kwargs["use_egoVlpV2_takeVideoFeats_usingCenterTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingCenterTime" in kwargs) else\
																False
		self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats = kwargs["maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats"]\
																					if ("maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats" in kwargs) else\
																						0

		self.use_egoVlpV2_takeVideoFeats = self.use_egoVlpV2_takeVideoFeats_usingStartNendTime or\
											self.use_egoVlpV2_takeVideoFeats_usingCenterTime

		self.use_videoLlama_feats = kwargs["use_videoLlama_feats"] if ("use_videoLlama_feats" in kwargs) else False
		self.videoLlama_feats_lenSeq = kwargs["videoLlama_feats_lenSeq"] if ("videoLlama_feats_lenSeq" in kwargs) else 32
		self.videoLlama_feats_seqAggregation = kwargs["videoLlama_feats_seqAggregation"] if ("videoLlama_feats_seqAggregation" in kwargs) else "cat"

		self.use_preExtractedFeats = self.use_egoVlpV2_takeVideoFeats or\
										self.use_videoLlama_feats

		self.num_classes = len(self.all_views)

		assert self.recog_arc in ['egovlp_v2']
		self.num_inFeats = None
		if self.recog_arc in ['egovlp_v2']:
			if self.use_videoLlama_feats:
				if self.videoLlama_feats_seqAggregation == "cat":
					self.num_inFeats = self.videoLlama_feats_lenSeq
				else:
					self.num_inFeats = 1
				self.num_inFeats *= 4096

			elif self.use_egoVlpV2_takeVideoFeats:
				if self.use_egoVlpV2_takeVideoFeats_usingStartNendTime:
					self.num_inFeats = (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1)
				elif self.use_egoVlpV2_takeVideoFeats_usingCenterTime:
					self.num_inFeats = 1
				self.num_inFeats *= 4096
			else:
				self.num_inFeats = 768

			self.transformer_pol = None
			self.pos_embed = None
			if self.use_transformerPol:
				encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_inFeats, nhead=8, dropout=self.transformerPol_dropout)
				self.transformer_pol = nn.TransformerEncoder(encoder_layer, num_layers=self.numLayers_transformerPol)

				if self.addPE_transformerPol:
					self.pos_embed = torch.from_numpy(get_1d_sincos_pos_embed(self.num_inFeats, len(self.all_views))).float()

			self.num_inFeats *= len(self.all_views)
			self.classifier = nn.Sequential()
			in_feats = self.num_inFeats
			for linearLayer_dim in self.linearLayer_dims:
				self.classifier.append(nn.Linear(in_feats, linearLayer_dim))
				self.classifier.append(nn.BatchNorm1d(linearLayer_dim))
				self.classifier.append(nn.ReLU(inplace=True))
				if self.linearLayer_dropout:
					self.classifier.append(nn.Dropout(self.linearLayer_dropout, inplace=False))
				in_feats = linearLayer_dim
			self.classifier.append(nn.Linear(in_feats, self.num_classes))

	def forward(self, feats, ):
		if self.use_preExtractedFeats:
			feats = feats.reshape((feats.shape[0], feats.shape[1] * feats.shape[2]))
		if self.use_videoLlama_feats:
			feats = feats.float()

		if self.transformer_pol is not None:
			feats = feats.reshape((feats.shape[0], len(self.all_views), -1)).permute((1, 0, 2))

			if self.pos_embed is not None:
				self.pos_embed = self.pos_embed.to(feats.device).unsqueeze(1).repeat((1, feats.shape[1], 1))
				feats = feats + self.pos_embed

			feats = self.transformer_pol(feats).permute((1, 0, 2))
			feats = feats.reshape((feats.shape[0], -1))

		out = self.classifier(feats)

		return out
