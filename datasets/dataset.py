import os
import pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2 as cv
from scipy.io import wavfile

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms._transforms_video import RandomCropVideo, RandomResizedCropVideo,CenterCropVideo, NormalizeVideo,ToTensorVideo,RandomHorizontalFlipVideo

import decord
from decord import VideoReader

from datasets.utils import frame_normalize
from common.utils import *
from common.dist_utils import *


def load_datapointVideo_egoExoNarrate(video_path,
										n_frms=8,
										height=-1,
										width=-1,
										sampling="uniform",
										dont_square_frames=False,):

	decord.bridge.set_bridge("torch")

	assert ospif(video_path), print(video_path)

	isEmpty_clp = False
	try:
		if dont_square_frames:
			vrs = VideoReader(uri=video_path, num_threads=1)
		else:
			assert height == width
			vrs = VideoReader(uri=video_path, height=height, width=width, num_threads=1)
	except:
		cap = cv.VideoCapture(video_path)
		assert int(cap.get(cv.CAP_PROP_FRAME_COUNT)) == 0
		cap.release()
		isEmpty_clp = True
	
	indices = None
	if not isEmpty_clp:
		vlen = len(vrs)
		start, end = 0, vlen

		orig_n_frms = n_frms
		n_frms = min(n_frms, vlen)

		if sampling == "uniform":
			indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
		elif sampling == "headtail":
			indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
			indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
			indices = indices_h + indices_t
		else:
		    raise NotImplementedError

		assert len(indices) >= 1
		assert len(indices) <= orig_n_frms
		if len(indices) < orig_n_frms:
			indices = indices + ([indices[-1]] * (orig_n_frms - len(indices)))
		assert len(indices) == orig_n_frms, print(len(indices), orig_n_frms, vlen, n_frms)

		""" get_batch -> T, H, W, C """
		temp_frms = vrs.get_batch(indices)	# vrs[indices], vrs.get_batch(indices), torch.stack([vrs[idx] for idx in indices])
		tensor_frms = torch.from_numpy(temp_frms) if (type(temp_frms) is not torch.Tensor) else temp_frms
	else:
		tensor_frms = torch.zeros((n_frms, height, width, 3))
		
	frms = tensor_frms.float() / 255  # .byte(), (T, H, W, C)

	return frms, indices


def get_rel_ce(ce1, 
				ce2, 
				return_coord_angles=False, 
				return_coord_normalized=False, 
				return_angles=False, 
				return_quarts=False, 
				return_onlyRotation=False):
	rot1 = np.array([ce1[0][:-1], ce1[1][:-1], ce1[2][:-1]])
	rot2 = np.array([ce2[0][:-1], ce2[1][:-1], ce2[2][:-1]])
	""" 2 wrt 1 """
	rot21 = np.matmul( rot1.T, rot2 )

	t1 = np.array([ce1[0][-1], ce1[1][-1], ce1[2][-1]])
	t2 = np.array([ce2[0][-1], ce2[1][-1], ce2[2][-1]])
	t21 = np.matmul(rot1.T, t2 - t1)

	if return_coord_angles:
		if t21[1] == t21[0] == 0:
			alpha = 0
		else:
			alpha = np.arctan(t21[1]/t21[0]) 
		
		beta = np.arcsin(t21[2] / (np.linalg.norm(t21) + 1e-13))

		alpha += np.pi
		alpha *= 180 / np.pi
		assert 0 <= alpha <= 360, print(alpha)
		if alpha == 360:
			alpha = 0
		assert 0 <= alpha < 360, print(alpha)

		beta += (np.pi / 2)
		beta *= 180 / np.pi
		assert 0 <= beta <= 180, print(beta, t21[2])

		t21 = [alpha, beta]
	elif return_coord_normalized:
		t21 = t21 / (np.linalg.norm(t21) + 1e-13)
	else:
		pass

	if return_angles:
		rot21_angls_zyx = Rotation.from_matrix(rot21).as_euler('zyx', degrees=False)

		ang_x, ang_y, ang_z = rot21_angls_zyx[2], rot21_angls_zyx[1], rot21_angls_zyx[0]

		ang_x += 180
		assert 0 <= ang_x <= 360, print(ang_x)
		if ang_x == 360:
			ang_x = 0
		assert 0 <= ang_x < 360, print(ang_x)

		ang_y += 90
		assert 0 <= ang_y <= 180, print(ang_y)

		ang_z += 180
		assert 0 <= ang_z <= 360, print(ang_z)
		if ang_z == 360:
			ang_z = 0
		assert 0 <= ang_z < 360, print(ang_z)

		ret = np.concatenate([t21, [ang_x, ang_y, ang_z]])
	elif return_quarts:
		rot21_quats_xyzw = Rotation.from_matrix(rot21).as_quat()
		ret = np.concatenate([t21, [rot21_quats_xyzw[0], rot21_quats_xyzw[1], rot21_quats_xyzw[2], rot21_quats_xyzw[3]]])
	else:
		ret = np.concatenate([t21, rot21[0], rot21[1], rot21[2]])

	if return_onlyRotation:
		if return_coord_angles:
			ret = ret[2:]
		else:
			ret = ret[3:]

	return ret


def compute_classWeights(num_views,
						lst_dtpnts,
						is_multiPseudolabler,
						topK_multiPseudolabler,
						bordaCount_multiPseudolabler,
						multiBestViewAggregator_multiPseudoLabler):
	bstIdx2cnt = {}
	for ele1 in tqdm(lst_dtpnts):
		if is_multiPseudolabler:
			if multiBestViewAggregator_multiPseudoLabler:
				assert len(ele1['scores']) == 3, print(ele1['scores'])
				bstIdx2cnt_thisEle = {}
				best_idxs_all = []
				for ele in ele1['scores']:
					best_idxs = []
					for bstVw_idx in np.where(ele == np.max(ele))[0].tolist():
						best_idxs.append(bstVw_idx)	
						if bstVw_idx not in bstIdx2cnt_thisEle:
							bstIdx2cnt_thisEle[bstVw_idx] = 0
						bstIdx2cnt_thisEle[bstVw_idx] += 1
					best_idxs_all.append(best_idxs)
				cnt2bstIdxs_thisEle = {}
				for bst_idx, cnt in bstIdx2cnt_thisEle.items():
					if cnt not in cnt2bstIdxs_thisEle:
						cnt2bstIdxs_thisEle[cnt] = set()
					cnt2bstIdxs_thisEle[cnt].add(bst_idx)
				cntNbstIdxs_thisEle = []
				for cnt, bst_idxs in cnt2bstIdxs_thisEle.items():
					cntNbstIdxs_thisEle.append((cnt, bst_idxs))
				srtd_cntNbstIdxs_thisEle = sorted(cntNbstIdxs_thisEle)[::-1]
				if srtd_cntNbstIdxs_thisEle[0][0] in [1]:
					best_idxs = best_idxs_all[0]
				else:
					best_idxs = list(srtd_cntNbstIdxs_thisEle[0][1])
			else:
				dtpnt_scrs_argsrt = np.argsort(np.array(ele1['scores']), axis=1)[:, ::-1]
				if bordaCount_multiPseudolabler:
					vwIdx2bordaCount = {}
					vwIdx2numVotes = {}
					for colIdx_dtpnt_scrs_argsrt in range(dtpnt_scrs_argsrt.shape[1]):
						col_dtpnt_scrs_argsrt = dtpnt_scrs_argsrt[:, colIdx_dtpnt_scrs_argsrt]
						col_bordaCount = dtpnt_scrs_argsrt.shape[1] - 1 - colIdx_dtpnt_scrs_argsrt
						for vw_idx in col_dtpnt_scrs_argsrt:
							if vw_idx not in vwIdx2bordaCount:
								vwIdx2bordaCount[vw_idx] = 0
								assert vw_idx not in vwIdx2numVotes
								vwIdx2numVotes[vw_idx] = 0

							vwIdx2bordaCount[vw_idx] += col_bordaCount
							vwIdx2numVotes[vw_idx] += 1

					lst_bordaCountNvwIdx = list()
					for vw_idx, total_bordaCount in vwIdx2bordaCount.items():
						assert  vwIdx2numVotes[vw_idx] == len(dtpnt_scrs_argsrt)
						lst_bordaCountNvwIdx.append((total_bordaCount, vw_idx))
					srtdLst_bordaCountNvwIdx = sorted(lst_bordaCountNvwIdx)[::-1]

					best_idxs = []
					for ele_srtdLst_bordaCountNvwIdx in srtdLst_bordaCountNvwIdx:
						if ele_srtdLst_bordaCountNvwIdx[0] == srtdLst_bordaCountNvwIdx[0][0]:
							best_idxs.append(ele_srtdLst_bordaCountNvwIdx[1])
						else:
							break
				else:
					dtpnt_scrs_argsrt_topK = dtpnt_scrs_argsrt[:, :topK_multiPseudolabler]	

					""" considers vote count and average rank"""
					idx2numVotesNtotalWeight = {}
					for row_dtpnt_scrs_argsrt_topK in dtpnt_scrs_argsrt_topK:
						for ele_idx, ele in enumerate(row_dtpnt_scrs_argsrt_topK):
							if ele not in idx2numVotesNtotalWeight:
								idx2numVotesNtotalWeight[ele] = [0, 0]

							idx2numVotesNtotalWeight[ele][0] += 1

							ele_weight =  topK_multiPseudolabler - ele_idx
							idx2numVotesNtotalWeight[ele][1] += ele_weight

					numVotes2avgWeightsNidxs = {}
					for vw_idx, num_votesNtotalWeight in idx2numVotesNtotalWeight.items():
						if num_votesNtotalWeight[0] not in numVotes2avgWeightsNidxs:
							numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]] = set()
						numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]].add((num_votesNtotalWeight[1] / max(num_votesNtotalWeight[0], 1), vw_idx ))
					lst_numVotesNavgWeightsNidxs = []
					for num_votes, avgWeightsNvwIdxs in numVotes2avgWeightsNidxs.items():
						lst_numVotesNavgWeightsNidxs.append((num_votes, list(avgWeightsNvwIdxs)))
					srtdLst_numVotesNavgWeightsNidxs = sorted(lst_numVotesNavgWeightsNidxs)[::-1]
					avgWeightsNidxs_wHighestVotes = srtdLst_numVotesNavgWeightsNidxs[0][1]

					srtdLst_avgWeightsNidxs_wHighestVotes = sorted(avgWeightsNidxs_wHighestVotes)[::-1]
					best_idxs = []
					for ele_srtdLst_avgWeightsNidxs_wHighestVotes in srtdLst_avgWeightsNidxs_wHighestVotes:
						if ele_srtdLst_avgWeightsNidxs_wHighestVotes[0] == srtdLst_avgWeightsNidxs_wHighestVotes[0][0]:
							best_idxs.append(ele_srtdLst_avgWeightsNidxs_wHighestVotes[1])
		else:
			best_idxs = []
			for bstVw_idx in np.where(ele1['scores'] == np.max(ele1['scores']))[0].tolist():
				best_idxs.append(bstVw_idx)	

		for bst_idx in best_idxs:
			if bst_idx not in bstIdx2cnt:
				bstIdx2cnt[bst_idx] = 0
			bstIdx2cnt[bst_idx] += 1 
	assert len(bstIdx2cnt) == num_views

	all_prcnts = []
	for k in sorted(list(bstIdx2cnt.keys())):
		all_prcnts.append(bstIdx2cnt[k] / max(np.sum(list(bstIdx2cnt.values())), 1) )

	return np.array(all_prcnts)


class train_dataset(object):
	def __init__(self, args, **kwargs):
		self.lemmaDataset_dct = pkl_ld("./data/lemma/v1/misc/"+\
                   "take__2__startNendEgoImageSuffix__2__timestamp_n_startNendClipName_n_startNendFrameIdx_n_listAtomicDescriptions_n_listImageSuffixes__train.pkl")

		self.args = args
		self.kwargs = kwargs
		self.distributed = kwargs["distributed"]
		self.epochs = kwargs["epochs"]
		self.num_samples = kwargs["num_trainSamples"]
		self.batch_size = kwargs["batch_size"]
		self.all_views = kwargs['all_views']
		self.num_frames = kwargs['num_frames']
		self.frame_height = kwargs['frame_height']
		self.frame_width = kwargs['frame_width']
		self.frame_horizontalFlip = kwargs["frame_horizontalFlip"] if ("frame_horizontalFlip" in kwargs) else False
		self.frame_colorJitter = kwargs["frame_colorJitter"] if ("frame_colorJitter" in kwargs) else [0, 0, 0]
		self.dont_square_frames = kwargs["dont_square_frames"] if ("dont_square_frames" in kwargs) else False
		self.videoClips_dir = kwargs["videoClips_dir"] if ("videoClips_dir" in kwargs) else None
		self.datapoint_videoClips_dir = kwargs["datapoint_videoClips_dir"] if ("datapoint_videoClips_dir" in kwargs) else None
		datapoints_filePath = kwargs["trainDatapoints_filePath"]
		datapoints_captioner_filePath = kwargs["trainDatapoints_captioner_filePath"] if ("trainDatapoints_captioner_filePath" in kwargs) else False
		self.recog_arc = kwargs['recog_arc']
		self.task_type = kwargs['task_type']
		self.randomize_trainLabel_forOneHot =\
			kwargs["randomize_trainLabel_forOneHot"] if ("randomize_trainLabel_forOneHot" in kwargs)\
				else False
		self.randomize_trainViewOrder = kwargs["randomize_trainViewOrder"] if ("randomize_trainViewOrder" in kwargs) else False
		self.use_datapointVideoClips = kwargs["use_datapointVideoClips"] if ("use_datapointVideoClips" in kwargs) else False

		self.use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
																	False
		self.use_egoVlpV2_takeVideoFeats_usingCenterTime = kwargs["use_egoVlpV2_takeVideoFeats_usingCenterTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingCenterTime" in kwargs) else\
																	False
		self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats = kwargs["maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats"]\
																					if ("maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats" in kwargs) else\
																						0
		self.padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
																				if ("padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
																					False
		self.egoVlpV2_takeVideoFeats_dir = kwargs["egoVlpV2_takeVideoFeats_dir"]\
											if ("egoVlpV2_takeVideoFeats_dir" in kwargs) else\
												None
		self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp = kwargs["egoVlpV2_takeVideoFeats_takeName2camId2featName_fp"]\
																	if ("egoVlpV2_takeVideoFeats_takeName2camId2featName_fp" in kwargs) else\
																		None	

		self.use_egoVlpV2_takeVideoFeats = self.use_egoVlpV2_takeVideoFeats_usingStartNendTime or\
											self.use_egoVlpV2_takeVideoFeats_usingCenterTime	

		self.use_videoLlama_feats = kwargs["use_videoLlama_feats"] if ("use_videoLlama_feats" in kwargs) else False
		self.videoLlama_feats_dir = kwargs["videoLlama_feats_dir"] if ("videoLlama_feats_dir" in kwargs) else None	
		self.videoLlama_feats_seqAggregation = kwargs["videoLlama_feats_seqAggregation"] if ("videoLlama_feats_seqAggregation" in kwargs) else "cat"									

		self.use_relativeCameraPoseLoss = kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False
		self.maskOut_invalidRelativeCameraPoseLoss_inTraining = kwargs["maskOut_invalidRelativeCameraPoseLoss_inTraining"] if ("maskOut_invalidRelativeCameraPoseLoss_inTraining" in kwargs) else False
		self.relativeCameraPoseLoss_rotationOnly = kwargs["relativeCameraPoseLoss_rotationOnly"] if ("relativeCameraPoseLoss_rotationOnly" in kwargs) else False
		self.relativeCameraPoseLoss_rotationInAngles = kwargs["relativeCameraPoseLoss_rotationInAngles"] if ("relativeCameraPoseLoss_rotationInAngles" in kwargs) else False
		self.relativeCameraPoseLoss_rotationInQuarts = kwargs["relativeCameraPoseLoss_rotationInQuarts"] if ("relativeCameraPoseLoss_rotationInQuarts" in kwargs) else False
		self.relativeCameraPoseLoss_coordsInAngles = kwargs["relativeCameraPoseLoss_coordsInAngles"] if ("relativeCameraPoseLoss_coordsInAngles" in kwargs) else False
		self.relativeCameraPoseLoss_coordsNormalized = kwargs["relativeCameraPoseLoss_coordsNormalized"] if ("relativeCameraPoseLoss_coordsNormalized" in kwargs) else False
		self.relativeCameraPoseLoss_refType = kwargs["relativeCameraPoseLoss_refType"] if ("relativeCameraPoseLoss_refType" in kwargs) else "first_view"
		self.relativeCameraPoseLoss_frameType = kwargs["relativeCameraPoseLoss_frameType"] if ("relativeCameraPoseLoss_frameType" in kwargs) else "all"
		self.cameraPose_dir = kwargs["cameraPose_dir"] if ("cameraPose_dir" in kwargs) else ""
		if self.use_relativeCameraPoseLoss:
			assert ospid(self.cameraPose_dir)

		self.isLemma_dataset = kwargs["isLemma_dataset"] if ("isLemma_dataset" in kwargs) else False

		self.total_num_samples = self.num_samples

		self.transforms = None
		if self.use_datapointVideoClips:
			assert self.frame_height == self.frame_width

			frame_mean, frame_std = frame_normalize(None, 
													input_frame_norm_type=self.recog_arc,
													return_meanNstd=True)
			frame_normalize_ = NormalizeVideo(mean=frame_mean, std=frame_std)

			trn_trnsfrms = [
			    transforms.Resize((self.frame_height)),
			    transforms.CenterCrop(self.frame_height),
			]

			if self.frame_horizontalFlip:
				trn_trnsfrms.append(RandomHorizontalFlipVideo())
			if (self.frame_colorJitter != [0, 0, 0]) and (self.frame_colorJitter != [0, 0]):
				if len(self.frame_colorJitter) == 2:
					self.transforms_colorJitter = transforms.ColorJitter(brightness=self.frame_colorJitter[0], 
																	saturation=self.frame_colorJitter[1],)
				else:
					self.transforms_colorJitter = transforms.ColorJitter(brightness=self.frame_colorJitter[0], 
																saturation=self.frame_colorJitter[1],
																hue=self.frame_colorJitter[2])
				self.transforms_normalize = frame_normalize_
			else:
				trn_trnsfrms.append(frame_normalize_)

			self.transforms = transforms.Compose(trn_trnsfrms)

		self.is_multiPseudolabler = False
		self.topK_multiPseudolabler = kwargs["topK_multiPseudolabler"] if ("topK_multiPseudolabler" in kwargs) else 1
		self.bordaCount_multiPseudolabler = kwargs["bordaCount_multiPseudolabler"] if ("bordaCount_multiPseudolabler" in kwargs) else False
		self.multiBestViewAggregator_multiPseudoLabler = kwargs["multiBestViewAggregator_multiPseudoLabler"] if ("multiBestViewAggregator_multiPseudoLabler" in kwargs) else False
		self.egoVlpV2_vis2textSim_labler = kwargs["egoVlpV2_vis2textSim_labler"] if ("egoVlpV2_vis2textSim_labler" in kwargs) else False
		assert 1 <= self.topK_multiPseudolabler <= len(self.all_views)
		if isinstance(datapoints_filePath, list):
			self.is_multiPseudolabler = True
			lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs = []
			for ele_datapoints_filePath in datapoints_filePath:
				assert isinstance(ele_datapoints_filePath, str)
				if ospif(ele_datapoints_filePath):
					with open(ele_datapoints_filePath, "rb") as fi:
						lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs.append(pickle.load(fi))
			self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs =\
				lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs[0]
		else:
			assert os.path.isfile(datapoints_filePath)
			with open(datapoints_filePath, "rb") as fi:
				self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs = pickle.load(fi)

		tkNm_2_strtNendTmstmp_cptnrScrs = None
		if self.egoVlpV2_vis2textSim_labler:
			assert isinstance(datapoints_captioner_filePath, str), print(datapoints_captioner_filePath)
			assert ospif(datapoints_captioner_filePath)
			tkNm_2_strtNendTmstmp_cptnrScrs = pkl_ld(datapoints_captioner_filePath)

		self.lst_dtpnts = []
		self.tkNm2cameraPose = {}
		for k1, v1 in tqdm(self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs.items()):
			if self.use_relativeCameraPoseLoss:
				if ospif(f"{self.cameraPose_dir}/{k1}.json"):
					with open(f"{self.cameraPose_dir}/{k1}.json", "r") as fi:
						cameraPose_thisTake = json.load(fi)
					assert k1 not in self.tkNm2cameraPose
					self.tkNm2cameraPose[k1] = cameraPose_thisTake
				else:
					if not self.maskOut_invalidRelativeCameraPoseLoss_inTraining:
						continue
 
			for k2, v2 in v1.items():
				if self.use_egoVlpV2_takeVideoFeats:
					if len(k2) == 3:
						assert isinstance(k2[1], (int, float))
						assert isinstance(k2[2], (int, float))
						dl_strtNendTmstmp = int(k2[2]) - int(k2[1])
					else:
						assert isinstance(k2[0], (int, float))
						assert isinstance(k2[1], (int, float))
						dl_strtNendTmstmp = int(k2[1]) - int(k2[0])
					assert dl_strtNendTmstmp >= 0, print(k2)
					if dl_strtNendTmstmp > self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats:
						continue

				if self.egoVlpV2_vis2textSim_labler:
					if len(v2['scores']) != 1:
						continue

				if self.egoVlpV2_vis2textSim_labler:
					self.lst_dtpnts.append({'take_name': k1,
											'startNend_clipName': v2['startNend_clipName'],
											'startNend_frameIdx': v2['startNend_frameIdx'],
											'startNend_timestamp': k2,})
				else:
					self.lst_dtpnts.append({'take_name': k1,
											'startNend_clipName': v2['startNend_clipName'],
											'startNend_frameIdx': v2['startNend_frameIdx'],
											'startNend_timestamp': k2,
											'timestamp': v2['timestamp']})

				if  self.isLemma_dataset:
					assert k1 in self.lemmaDataset_dct
					assert (k2[1], k2[2]) in self.lemmaDataset_dct[k1], print(k2, (k2[1], k2[2]), list(self.lemmaDataset_dct[k1].keys())[:2])
					self.lst_dtpnts[-1]["list_egoNexoSuffixes"] = self.lemmaDataset_dct[k1][(k2[1], k2[2])]["list_egoNexoSuffixes"]
				else:
					if len(k2) == 2:
						if ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[0]}_{k2[1]}.mp4"):
							pass
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[0])}_{k2[1]}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (int(k2[0]), k2[1])
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[0]}_{int(k2[1])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = ( k2[0], int(k2[1]))
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[0])}_{int(k2[1])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (int(k2[0]), int(k2[1]))
						else:
							raise ValueError
					else:
						if ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[1]}_{k2[2]}.mp4"):
							pass
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[1])}_{k2[2]}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], int(k2[1]), k2[2])
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[1]}_{int(k2[2])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], k2[1], int(k2[2]))
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[1])}_{int(k2[2])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], int(k2[1]), int(k2[2]))
						else:
							raise ValueError

				if self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
					if self.is_multiPseudolabler:
						raise ValueError
					self.lst_dtpnts[-1]['best_exo_views'] = v2['best_exo_views']
				else:
					if self.is_multiPseudolabler:
						self.lst_dtpnts[-1]['scores'] = []
						for ele__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs in\
								lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs:
							self.lst_dtpnts[-1]['scores'].append(
										ele__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs[k1][k2]['scores']
									)
							if self.all_views == ['1', '2', '3', '4', ]:
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][1:]
							elif self.all_views in [['aria', '1'], ['aria', '2'], ['aria', '3'], ['aria', '4']]:
								exo_strtIdx = int(self.all_views[-1])
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][:1] +\
																	self.lst_dtpnts[-1]['scores'][-1][exo_strtIdx: (exo_strtIdx + 1)]
					else:
						if self.egoVlpV2_vis2textSim_labler:
							if self.all_views != ['aria', '1', '2', '3', '4', ]:
								raise NotImplementedError
							self.lst_dtpnts[-1]['scores'] = v2['scores'][0]
							self.lst_dtpnts[-1]['scores_captioner'] = tkNm_2_strtNendTmstmp_cptnrScrs[k1][(round(k2[0], 4), round(k2[1], 4), round(k2[2], 4))]['scores']# , print(tkNm_2_strtNendTmstmp_cptnrScrs[k1].keys())
						else:
							self.lst_dtpnts[-1]['scores'] = v2['scores']
							if self.all_views == ['1', '2', '3', '4',]:
								self.lst_dtpnts[-1]['scores'] = self.lst_dtpnts[-1]['scores'][1:]
							elif self.all_views in [['aria', '1'], ['aria', '2'], ['aria', '3'], ['aria', '4']]:
								exo_strtIdx = int(self.all_views[-1])
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][:1] +\
																	self.lst_dtpnts[-1]['scores'][-1][exo_strtIdx: (exo_strtIdx + 1)]

		self.egoVlpV2_takeVideoFeats_takeName2camId2featName = None
		if self.use_egoVlpV2_takeVideoFeats:
			assert ospid(self.egoVlpV2_takeVideoFeats_dir)

			assert ospif(self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp)
			self.egoVlpV2_takeVideoFeats_takeName2camId2featName = pkl_ld(self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp)

		if self.task_type != "classify_oneHot_bestExoPred":
			self.class_weights = compute_classWeights(len(self.all_views),
													  self.lst_dtpnts,
													  self.is_multiPseudolabler,
													  self.topK_multiPseudolabler,
													  self.bordaCount_multiPseudolabler,
													  self.multiBestViewAggregator_multiPseudoLabler,
													  )
		else:
			self.class_weights = np.array([0, 0, 0, 0])

	def __len__(self):
		return self.total_num_samples	# 2, 6, 24, self.total_num_samples

	def __getitem__(self, index):
		dtpnt_idx = torch.randint(len(self.lst_dtpnts), (1,)).item()
		dtpnt = self.lst_dtpnts[dtpnt_idx]
		al_frms = []
		al_cameraPoses = None
		if self.use_relativeCameraPoseLoss:
			al_cameraPoses = []
		for vw in self.all_views:
			tk_nm = dtpnt['take_name']
			if len(dtpnt['startNend_timestamp']) == 3:
				cntr_tmstmp = dtpnt['startNend_timestamp'][0]
				strt_tmstmp = dtpnt['startNend_timestamp'][1]
				end_tmstmp = dtpnt['startNend_timestamp'][2]
			else:
				cntr_tmstmp = dtpnt['timestamp'] if ('timestamp' in dtpnt) else None 
				strt_tmstmp = dtpnt['startNend_timestamp'][0]
				end_tmstmp = dtpnt['startNend_timestamp'][1]
			if self.use_egoVlpV2_takeVideoFeats:
				if self.isLemma_dataset:
					raise NotImplementedError
				if self.use_relativeCameraPoseLoss:
					raise NotImplementedError

				assert tk_nm in self.egoVlpV2_takeVideoFeats_takeName2camId2featName
				egoVlpV2_takeVideoFeats_camId2featName = self.egoVlpV2_takeVideoFeats_takeName2camId2featName[tk_nm]

				assert vw in egoVlpV2_takeVideoFeats_camId2featName
				ft_nm = egoVlpV2_takeVideoFeats_camId2featName[vw]

				ft_fp = f"{self.egoVlpV2_takeVideoFeats_dir}/{ft_nm}"
				ft = torch.load(ft_fp, map_location="cpu")

				srt_tmstmp_int = int(strt_tmstmp)
				end_tmstmp_int = int(end_tmstmp)
				assert 0 <= end_tmstmp_int - srt_tmstmp_int <= self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats

				if self.use_egoVlpV2_takeVideoFeats_usingCenterTime:
					srt_tmstmp_int = end_tmstmp_int = int(cntr_tmstmp)
				
				frms = None
				ft_slc = None
				tmstmp_cnt = 0
				for tmstmp_int in range(srt_tmstmp_int, end_tmstmp_int + 1):
					if tmstmp_int < len(ft):
						ft_slc = ft[tmstmp_int]
						if frms is None:
							frms = ft_slc
						else:
							frms = torch.cat((frms, ft_slc))
					else:
						break
					tmstmp_cnt += 1

				if frms is None:
					assert ft_slc is None
					ft_slc = torch.zeros(4096)
					frms = torch.zeros(0)
				else:
					assert ft_slc is not None

				if self.use_egoVlpV2_takeVideoFeats_usingCenterTime:
					frms = ft_slc
					assert len(frms) == 4096
				elif self.use_egoVlpV2_takeVideoFeats_usingStartNendTime:
					if self.padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime:
						frms = torch.cat((frms, 
											torch.zeros(4096 * (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1 - tmstmp_cnt))))
					else:
						frms = torch.cat([frms] +\
											([ft_slc] * (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1 - tmstmp_cnt)))
					assert len(frms) == (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1) * 4096
			elif self.use_videoLlama_feats:
				if self.isLemma_dataset:
					raise NotImplementedError
				if self.use_relativeCameraPoseLoss:
					raise NotImplementedError

				strt_clpNm = dtpnt['startNend_clipName'][0]
				end_clpNm = dtpnt['startNend_clipName'][1]
				strt_frmIdx = dtpnt['startNend_frameIdx'][0]
				end_frmIdx = dtpnt['startNend_frameIdx'][1]

				ft_fp = f"{self.videoLlama_feats_dir}/{vw}/{tk_nm}/{strt_clpNm}_{end_clpNm}__{strt_frmIdx}_{end_frmIdx}.pt"
				assert ospif(ft_fp), print(ft_fp)

				ft = torch.load(ft_fp, map_location="cpu")
				if self.videoLlama_feats_seqAggregation == "mean":
					ft = torch.mean(ft, dim=0).unsqueeze(0)
				frms = ft.reshape((ft.shape[0] * ft.shape[1]))
			else:
				if self.use_datapointVideoClips:

					if self.isLemma_dataset:
						lst_egoNexoSffxs = dtpnt['list_egoNexoSuffixes']
						lst_imgSffxs = []
						for egoNexo_sffx in lst_egoNexoSffxs:
							if vw == "fpv1":
								lst_imgSffxs.append(egoNexo_sffx[0])
							elif vw == "master":
								lst_imgSffxs.append(egoNexo_sffx[1])
							else:
								raise ValueError

						assert self.frame_height == self.frame_width

						frms = []
						for img_sffx in lst_imgSffxs:
							img_fp = f"{self.datapoint_videoClips_dir}/{img_sffx}"
							assert os.path.isfile(img_fp)
							tmp_img = np.array(Image.open(img_fp).resize((self.frame_height, self.frame_width), resample=Image.BICUBIC))
							assert tmp_img.dtype == np.uint8
							frms.append(torch.from_numpy(tmp_img))

						frms = torch.stack(frms).float() / 255

					else:
						strt_clpNm = dtpnt['startNend_clipName'][0]
						end_clpNm = dtpnt['startNend_clipName'][1]
						strt_frmIdx = dtpnt['startNend_frameIdx'][0]
						end_frmIdx = dtpnt['startNend_frameIdx'][1]

						clp_pth = f"{self.datapoint_videoClips_dir}/{vw}/{tk_nm}/"+\
									f"{strt_clpNm}_{end_clpNm}__{strt_frmIdx}_{end_frmIdx}__{strt_tmstmp}_{end_tmstmp}.mp4"

						frms, frm_idxs = load_datapointVideo_egoExoNarrate(clp_pth,
																			n_frms=self.num_frames,
																			height=self.frame_height,
																			width=self.frame_width,
																			dont_square_frames=self.dont_square_frames,)
					if self.use_relativeCameraPoseLoss:
						if self.isLemma_dataset:
							raise NotImplementedError
						relativeCameraPoseLoss_global_frameIdxs = []

						if tk_nm in self.tkNm2cameraPose:
							has_cameraPose = True
							cameraPose_thisTake = self.tkNm2cameraPose[tk_nm]
						else:
							has_cameraPose = False
							cameraPose_thisTake = self.tkNm2cameraPose[list(self.tkNm2cameraPose.keys())[0]]

						if self.relativeCameraPoseLoss_frameType == "center":
							if has_cameraPose:
								relativeCameraPoseLoss_frameIdx = frm_idxs[len(frm_idxs) // 2]
								relativeCameraPoseLoss_global_frameIdxs = [(int(strt_clpNm) * 900) + int(strt_frmIdx) + int(relativeCameraPoseLoss_frameIdx)]
							else:
								relativeCameraPoseLoss_global_frameIdxs = [0]
						elif self.relativeCameraPoseLoss_frameType == "all":
							if has_cameraPose:						
								for relativeCameraPoseLoss_frameIdx in frm_idxs:
									relativeCameraPoseLoss_global_frameIdx = (int(strt_clpNm) * 900) + int(strt_frmIdx) + int(relativeCameraPoseLoss_frameIdx)
									relativeCameraPoseLoss_global_frameIdxs.append(relativeCameraPoseLoss_global_frameIdx)
							else:
								relativeCameraPoseLoss_global_frameIdxs = [0] * len(frm_idxs)
						else:
							raise NotImplementedError

						cameraPoses_thisVw = []
						if vw == 'aria':
							assert 'ego' in cameraPose_thisTake

							for relativeCameraPoseLoss_global_frameIdx in relativeCameraPoseLoss_global_frameIdxs:
								assert str(relativeCameraPoseLoss_global_frameIdx) in cameraPose_thisTake['ego']
								cameraPose_thisVw = cameraPose_thisTake['ego'][str(relativeCameraPoseLoss_global_frameIdx)]
								cameraPoses_thisVw.append(cameraPose_thisVw)

						else:
							assert vw in cameraPose_thisTake

							for relativeCameraPoseLoss_global_frameIdx in relativeCameraPoseLoss_global_frameIdxs:
								cameraPose_thisVw = cameraPose_thisTake[vw]
								cameraPoses_thisVw.append(cameraPose_thisVw)

						al_cameraPoses.append(cameraPoses_thisVw)

					frms = frms.permute(3, 0, 1, 2) # (T, H, W, C -> C, T, H, W)
					assert self.transforms is not None
					frms = self.transforms(frms)
					frms = frms.permute(1, 2, 3, 0)
				else:
					if self.isLemma_dataset:
						raise NotImplementedError
					if self.use_relativeCameraPoseLoss:
						raise NotImplementedError

					clp_dr = f"{self.videoClips_dir}/{vw}/{tk_nm}"
					assert os.path.isdir(clp_dr)
					frms = load_video_egoExoNarrate(clp_dr,
													dtpnt['startNend_clipName'],
													dtpnt['startNend_frameIdx'],
													n_frms=self.num_frames,
													height=self.frame_height,
													width=self.frame_width)
					frms = frame_normalize(frms, input_frame_norm_type=self.recog_arc)

			al_frms.append(frms)	

		ref_camerPoses = None
		if self.use_relativeCameraPoseLoss:
			ref_camerPoses = al_cameraPoses[0]

		class_weights = self.class_weights.copy()
		if self.randomize_trainViewOrder:
			vw_idxs = torch.randperm(len(al_frms)).tolist()

			al_frms_nw = []
			al_cameraPoses_nw = []
			for vw_idx in vw_idxs:
				al_frms_nw.append(al_frms[vw_idx])
				if self.use_relativeCameraPoseLoss:
					al_cameraPoses_nw.append(al_cameraPoses[vw_idx])

			al_frms = al_frms_nw
			al_cameraPoses = al_cameraPoses_nw

			if self.task_type in ["classify_oneHot", "match_dist"]:
				if self.is_multiPseudolabler:
					dtpnt_scrs = np.array(dtpnt['scores'])[:, vw_idxs]
				else:
					dtpnt_scrs = np.array(dtpnt['scores'])[vw_idxs]
					if self.egoVlpV2_vis2textSim_labler:
						dtpnt_cptnr_scrs = np.array(dtpnt['scores_captioner'])[vw_idxs]
			else:
				raise NotImplementedError

			class_weights = class_weights[vw_idxs]
		else:
			if self.task_type in ["classify_oneHot", "match_dist"]:
				dtpnt_scrs = dtpnt['scores']
			else:
				dtpnt_scrs = dtpnt['best_exo_views']

			if self.egoVlpV2_vis2textSim_labler:
				dtpnt_cptnr_scrs = dtpnt['scores_captioner']

		if isinstance(dtpnt_scrs[0], str):
			dtpnt_scrs_tmp = [0, 0, 0, 0]
			dtpnt_scrs_tmp[int(dtpnt_scrs[0]) - 1] = 1
			dtpnt_scrs_tnsr = torch.tensor(dtpnt_scrs_tmp).float()
		else:
			dtpnt_scrs_tnsr = torch.tensor(dtpnt_scrs).float()
		if len(dtpnt_scrs_tnsr.shape) == 1:
			dtpnt_scrs_tnsr = dtpnt_scrs_tnsr.unsqueeze(0)

		if self.egoVlpV2_vis2textSim_labler:
			dtpnt_cptnr_scrs_tnsr = torch.tensor(dtpnt_cptnr_scrs).float()
			if len(dtpnt_cptnr_scrs_tnsr.shape) == 1:
				dtpnt_cptnr_scrs_tnsr = dtpnt_cptnr_scrs_tnsr.unsqueeze(0)

		al_frms = torch.stack(al_frms)
		class_weights = torch.from_numpy(class_weights).float()

		al_rel_cameraPoses = None
		if self.use_relativeCameraPoseLoss:
			if self.relativeCameraPoseLoss_refType == "first_view":
				ref_camerPoses = al_cameraPoses[0]

			al_rel_cameraPoses = []
			for camera_poses in al_cameraPoses:
				if self.relativeCameraPoseLoss_refType == "first_view":
					rel_cameraPoses_thisVw = []
					for camerPose_idx, camera_pose in enumerate(camera_poses):
						rel_cameraPose = get_rel_ce(ref_camerPoses[camerPose_idx], camera_pose,
													return_angles=self.relativeCameraPoseLoss_rotationInAngles,
													return_quarts=self.relativeCameraPoseLoss_rotationInQuarts,
													return_onlyRotation=self.relativeCameraPoseLoss_rotationOnly,
													return_coord_angles=self.relativeCameraPoseLoss_coordsInAngles,
													return_coord_normalized=self.relativeCameraPoseLoss_coordsNormalized,)
						rel_cameraPoses_thisVw.append(rel_cameraPose)

					al_rel_cameraPoses.append(rel_cameraPoses_thisVw)
				elif self.relativeCameraPoseLoss_refType == "all_views":
					for camera_poses2 in al_cameraPoses:
						rel_cameraPoses_thisVwPr = []
						for camerPose_idx, camera_pose in enumerate(camera_poses2):
							rel_cameraPose = get_rel_ce(camera_poses[camerPose_idx], camera_pose,
														return_angles=self.relativeCameraPoseLoss_rotationInAngles,
														return_quarts=self.relativeCameraPoseLoss_rotationInQuarts,
														return_onlyRotation=self.relativeCameraPoseLoss_rotationOnly,
														return_coord_angles=self.relativeCameraPoseLoss_coordsInAngles,
														return_coord_normalized=self.relativeCameraPoseLoss_coordsNormalized,)
							rel_cameraPoses_thisVwPr.append(rel_cameraPose)

						al_rel_cameraPoses.append(rel_cameraPoses_thisVwPr)

			al_rel_cameraPoses = torch.from_numpy(np.array(al_rel_cameraPoses)).float() # np.concatenate -> np.array

		if self.task_type in ["classify_oneHot", "match_dist"]:
			if self.task_type == "classify_oneHot":
				if self.is_multiPseudolabler:
					if self.multiBestViewAggregator_multiPseudoLabler:
						assert len(dtpnt_scrs) == 3
						bstIdx2cnt_thisEle = {}
						best_idxs_all = []
						for dtpnt_scrs_ele in dtpnt_scrs:
							best_idxs = []
							for bstVw_idx in np.where(dtpnt_scrs_ele == np.max(dtpnt_scrs_ele))[0].tolist():
								best_idxs.append(bstVw_idx)	
								if bstVw_idx not in bstIdx2cnt_thisEle:
									bstIdx2cnt_thisEle[bstVw_idx] = 0
								bstIdx2cnt_thisEle[bstVw_idx] += 1
							best_idxs_all.append(best_idxs)
						cnt2bstIdxs_thisEle = {}
						for bst_idx, cnt in bstIdx2cnt_thisEle.items():
							if cnt not in cnt2bstIdxs_thisEle:
								cnt2bstIdxs_thisEle[cnt] = set()
							cnt2bstIdxs_thisEle[cnt].add(bst_idx)
						cntNbstIdxs_thisEle = []
						for cnt, bst_idxs in cnt2bstIdxs_thisEle.items():
							cntNbstIdxs_thisEle.append((cnt, bst_idxs))
						srtd_cntNbstIdxs_thisEle = sorted(cntNbstIdxs_thisEle)[::-1]
						if srtd_cntNbstIdxs_thisEle[0][0] in [1]:
							best_idxs = best_idxs_all[0]
						else:
							best_idxs = list(srtd_cntNbstIdxs_thisEle[0][1])
					else:
						dtpnt_scrs_argsrt = np.argsort(dtpnt_scrs, axis=1)[:, ::-1]
						if self.bordaCount_multiPseudolabler:
							vwIdx2bordaCount = {}
							vwIdx2numVotes = {}
							for colIdx_dtpnt_scrs_argsrt in range(dtpnt_scrs_argsrt.shape[1]):
								col_dtpnt_scrs_argsrt = dtpnt_scrs_argsrt[:, colIdx_dtpnt_scrs_argsrt]
								col_bordaCount = dtpnt_scrs_argsrt.shape[1] - 1 - colIdx_dtpnt_scrs_argsrt
								for vw_idx in col_dtpnt_scrs_argsrt:
									if vw_idx not in vwIdx2bordaCount:
										vwIdx2bordaCount[vw_idx] = 0
										assert vw_idx not in vwIdx2numVotes
										vwIdx2numVotes[vw_idx] = 0

									vwIdx2bordaCount[vw_idx] += col_bordaCount
									vwIdx2numVotes[vw_idx] += 1

							lst_bordaCountNvwIdx = list()
							for vw_idx, total_bordaCount in vwIdx2bordaCount.items():
								assert  vwIdx2numVotes[vw_idx] == len(dtpnt_scrs_argsrt)
								lst_bordaCountNvwIdx.append((total_bordaCount, vw_idx))
							srtdLst_bordaCountNvwIdx = sorted(lst_bordaCountNvwIdx)[::-1]

							best_idxs = []
							for ele_srtdLst_bordaCountNvwIdx in srtdLst_bordaCountNvwIdx:
								if ele_srtdLst_bordaCountNvwIdx[0] == srtdLst_bordaCountNvwIdx[0][0]:
									best_idxs.append(ele_srtdLst_bordaCountNvwIdx[1])
								else:
									break
						else:
							dtpnt_scrs_argsrt_topK = dtpnt_scrs_argsrt[:, :self.topK_multiPseudolabler]	

							""" considers vote count and average rank"""
							idx2numVotesNtotalWeight = {}
							for row_dtpnt_scrs_argsrt_topK in dtpnt_scrs_argsrt_topK:
								for ele_idx, ele in enumerate(row_dtpnt_scrs_argsrt_topK):
									if ele not in idx2numVotesNtotalWeight:
										idx2numVotesNtotalWeight[ele] = [0, 0]

									idx2numVotesNtotalWeight[ele][0] += 1

									ele_weight =  self.topK_multiPseudolabler - ele_idx
									idx2numVotesNtotalWeight[ele][1] += ele_weight

							numVotes2avgWeightsNidxs = {}
							for vw_idx, num_votesNtotalWeight in idx2numVotesNtotalWeight.items():
								if num_votesNtotalWeight[0] not in numVotes2avgWeightsNidxs:
									numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]] = set()
								numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]].add((num_votesNtotalWeight[1] / max(num_votesNtotalWeight[0], 1), vw_idx ))
							lst_numVotesNavgWeightsNidxs = []
							for num_votes, avgWeightsNvwIdxs in numVotes2avgWeightsNidxs.items():
								lst_numVotesNavgWeightsNidxs.append((num_votes, list(avgWeightsNvwIdxs)))
							srtdLst_numVotesNavgWeightsNidxs = sorted(lst_numVotesNavgWeightsNidxs)[::-1]
							avgWeightsNidxs_wHighestVotes = srtdLst_numVotesNavgWeightsNidxs[0][1]

							srtdLst_avgWeightsNidxs_wHighestVotes = sorted(avgWeightsNidxs_wHighestVotes)[::-1]
							best_idxs = []
							for ele_srtdLst_avgWeightsNidxs_wHighestVotes in srtdLst_avgWeightsNidxs_wHighestVotes:
								if ele_srtdLst_avgWeightsNidxs_wHighestVotes[0] == srtdLst_avgWeightsNidxs_wHighestVotes[0][0]:
									best_idxs.append(ele_srtdLst_avgWeightsNidxs_wHighestVotes[1])
					if self.randomize_trainLabel_forOneHot:
						lbl_idx = best_idxs[torch.randint(len(best_idxs), size=(1,)).item()]
					else:
						lbl_idx = best_idxs[0]
				else:
					if self.randomize_trainLabel_forOneHot:
						lbl_idx = np.where(dtpnt_scrs == np.max(dtpnt_scrs))[0].tolist()[torch.randint(len(np.where(dtpnt_scrs == np.max(dtpnt_scrs))[0]), size=(1,)).item()]
					else:
						lbl_idx = np.argmax(dtpnt_scrs)
				lbl = torch.tensor([lbl_idx]).long()
			elif self.task_type == "match_dist":
				if self.is_multiPseudolabler:
					print("need to normalize scores")
					raise NotImplementedError
					dtpnt_scrs = np.mean(dtpnt_scrs, axis=0)
				lbl = torch.tensor(dtpnt_scrs).float()

			lbl_multiHot = torch.zeros((len(self.all_views))).float()
			if (self.task_type in ["classify_oneHot"]) and (self.is_multiPseudolabler):
				for bstVw_idx in best_idxs:
					lbl_multiHot[bstVw_idx] = 1	
			else:
				for bstVw_idx in np.where(dtpnt_scrs == np.max(dtpnt_scrs))[0].tolist():
					lbl_multiHot[bstVw_idx] = 1	

		elif self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
			if self.randomize_trainViewOrder:
				raise NotImplementedError

			lbl = torch.zeros((len(self.all_views))).float()
			for bst_ex_vw in dtpnt_scrs:
				lbl[int(bst_ex_vw) - 1] = 1	

			if self.task_type == "classify_oneHot_bestExoPred":
				lbl_multiHot = lbl
				if self.randomize_trainLabel_forOneHot:
					lbl = torch.tensor([int(dtpnt_scrs[torch.randint(len(dtpnt_scrs), size=(1,)).item()])]).long() - 1	
				else:
					lbl = torch.tensor([int(dtpnt_scrs[0])]).long() - 1	
		else:
			raise NotImplementedError

		if self.task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
			if self.use_relativeCameraPoseLoss:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
				else:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
			else:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights
				else:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, class_weights
		else:
			if self.use_relativeCameraPoseLoss:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
				else:
					return al_frms, lbl, dtpnt_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
			else:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights
				else:
					return al_frms, lbl, dtpnt_scrs_tnsr, class_weights


class val_dataset(object):
	def __init__(self, args, **kwargs):
		self.lemmaDataset_dct =\
			pkl_ld("./data/lemma/v1/misc/"+\
				  "take__2__startNendEgoImageSuffix__2__timestamp_n_startNendClipName_n_startNendFrameIdx_n_listAtomicDescriptions_n_listImageSuffixes__val.pkl")

		self.args = args
		self.kwargs = kwargs
		self.distributed = kwargs["distributed"]
		self.epochs = kwargs["epochs"]
		self.num_samples = kwargs["num_valSamples"]
		self.batch_size = kwargs["batch_size"]
		self.all_views = kwargs['all_views']
		self.num_frames = kwargs['num_frames']
		self.frame_height = kwargs['frame_height']
		self.frame_width = kwargs['frame_width']
		self.dont_square_frames = kwargs["dont_square_frames"] if ("dont_square_frames" in kwargs) else False
		self.videoClips_dir = kwargs["videoClips_dir"] if ("videoClips_dir" in kwargs) else None
		self.datapoint_videoClips_dir = kwargs["datapoint_videoClips_dir"] if ("datapoint_videoClips_dir" in kwargs) else None
		datapoints_filePath = kwargs["valDatapoints_filePath"]
		datapoints_captioner_filePath = kwargs["valDatapoints_captioner_filePath"] if ("valDatapoints_captioner_filePath" in kwargs) else None
		self.recog_arc = kwargs['recog_arc']
		self.task_type = kwargs['task_type']
		self.use_datapointVideoClips = kwargs["use_datapointVideoClips"] if ("use_datapointVideoClips" in kwargs) else False
		self.use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
																	False
		self.use_egoVlpV2_takeVideoFeats_usingCenterTime = kwargs["use_egoVlpV2_takeVideoFeats_usingCenterTime"]\
																if ("use_egoVlpV2_takeVideoFeats_usingCenterTime" in kwargs) else\
																	False
		self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats = kwargs["maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats"]\
																					if ("maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats" in kwargs) else\
																						0
		self.padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
																				if ("padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
																					False
		self.egoVlpV2_takeVideoFeats_dir = kwargs["egoVlpV2_takeVideoFeats_dir"]\
											if ("egoVlpV2_takeVideoFeats_dir" in kwargs) else\
												None
		self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp = kwargs["egoVlpV2_takeVideoFeats_takeName2camId2featName_fp"]\
																	if ("egoVlpV2_takeVideoFeats_takeName2camId2featName_fp" in kwargs) else\
																		None

		self.use_egoVlpV2_takeVideoFeats = self.use_egoVlpV2_takeVideoFeats_usingStartNendTime or\
											self.use_egoVlpV2_takeVideoFeats_usingCenterTime

		self.use_videoLlama_feats = kwargs["use_videoLlama_feats"] if ("use_videoLlama_feats" in kwargs) else False
		self.videoLlama_feats_dir = kwargs["videoLlama_feats_dir"] if ("videoLlama_feats_dir" in kwargs) else None	
		self.videoLlama_feats_seqAggregation = kwargs["videoLlama_feats_seqAggregation"] if ("videoLlama_feats_seqAggregation" in kwargs) else "cat"	

		self.use_relativeCameraPoseLoss = kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False
		self.relativeCameraPoseLoss_rotationOnly = kwargs["relativeCameraPoseLoss_rotationOnly"] if ("relativeCameraPoseLoss_rotationOnly" in kwargs) else False
		self.relativeCameraPoseLoss_rotationInAngles = kwargs["relativeCameraPoseLoss_rotationInAngles"] if ("relativeCameraPoseLoss_rotationInAngles" in kwargs) else False
		self.relativeCameraPoseLoss_rotationInQuarts = kwargs["relativeCameraPoseLoss_rotationInQuarts"] if ("relativeCameraPoseLoss_rotationInQuarts" in kwargs) else False
		self.relativeCameraPoseLoss_coordsInAngles = kwargs["relativeCameraPoseLoss_coordsInAngles"] if ("relativeCameraPoseLoss_coordsInAngles" in kwargs) else False
		self.relativeCameraPoseLoss_coordsNormalized = kwargs["relativeCameraPoseLoss_coordsNormalized"] if ("relativeCameraPoseLoss_coordsNormalized" in kwargs) else False
		self.relativeCameraPoseLoss_refType = kwargs["relativeCameraPoseLoss_refType"] if ("relativeCameraPoseLoss_refType" in kwargs) else "first_view"
		self.relativeCameraPoseLoss_frameType = kwargs["relativeCameraPoseLoss_frameType"] if ("relativeCameraPoseLoss_frameType" in kwargs) else "all"
		self.cameraPose_dir = kwargs["cameraPose_dir"] if ("cameraPose_dir" in kwargs) else ""
		if self.use_relativeCameraPoseLoss:
			assert ospid(self.cameraPose_dir)

		self.isLemma_dataset = kwargs["isLemma_dataset"] if ("isLemma_dataset" in kwargs) else False

		self.total_num_samples = self.num_samples

		self.transforms = None
		if self.use_datapointVideoClips:
			assert self.frame_height == self.frame_width

			frame_mean, frame_std = frame_normalize(None, 
													input_frame_norm_type=self.recog_arc,
													return_meanNstd=True)
			frame_normalize_ = NormalizeVideo(mean=frame_mean, std=frame_std)

			self.transforms = transforms.Compose([
			    transforms.Resize(self.frame_height),
			    transforms.CenterCrop(self.frame_height),
			    frame_normalize_,
			])

		self.is_multiPseudolabler = False
		self.topK_multiPseudolabler = kwargs["topK_multiPseudolabler"] if ("topK_multiPseudolabler" in kwargs) else 1
		self.bordaCount_multiPseudolabler = kwargs["bordaCount_multiPseudolabler"] if ("bordaCount_multiPseudolabler" in kwargs) else False
		self.multiBestViewAggregator_multiPseudoLabler = kwargs["multiBestViewAggregator_multiPseudoLabler"] if ("multiBestViewAggregator_multiPseudoLabler" in kwargs) else False
		self.egoVlpV2_vis2textSim_labler = kwargs["egoVlpV2_vis2textSim_labler"] if ("egoVlpV2_vis2textSim_labler" in kwargs) else False
		assert 1 <= self.topK_multiPseudolabler <= len(self.all_views)
		if isinstance(datapoints_filePath, list):
			self.is_multiPseudolabler = True
			lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs = []
			for ele_datapoints_filePath in datapoints_filePath:
				assert isinstance(ele_datapoints_filePath, str)
				if ospif(ele_datapoints_filePath):
					with open(ele_datapoints_filePath, "rb") as fi:
						lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs.append(pickle.load(fi))
			self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs =\
				lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs[0]
		else:
			assert os.path.isfile(datapoints_filePath)
			with open(datapoints_filePath, "rb") as fi:
				self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs = pickle.load(fi)

		tkNm_2_strtNendTmstmp_cptnrScrs = None
		if self.egoVlpV2_vis2textSim_labler:
			assert isinstance(datapoints_captioner_filePath, str), print(datapoints_captioner_filePath)
			assert ospif(datapoints_captioner_filePath)
			tkNm_2_strtNendTmstmp_cptnrScrs = pkl_ld(datapoints_captioner_filePath)

		self.lst_dtpnts = []
		self.tkNm2cameraPose = {}
		for k1, v1 in tqdm(self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs.items()):
			if self.use_relativeCameraPoseLoss:
				if ospif(f"{self.cameraPose_dir}/{k1}.json"):
					with open(f"{self.cameraPose_dir}/{k1}.json", "r") as fi:
						cameraPose_thisTake = json.load(fi)

					assert k1 not in self.tkNm2cameraPose
					self.tkNm2cameraPose[k1] = cameraPose_thisTake

			for k2, v2 in v1.items():
				if self.use_egoVlpV2_takeVideoFeats:
					if len(k2) == 3:
						assert isinstance(k2[1], (int, float))
						assert isinstance(k2[2], (int, float))
						dl_strtNendTmstmp = int(k2[2]) - int(k2[1])
					else:
						assert isinstance(k2[0], (int, float))
						assert isinstance(k2[1], (int, float))
						dl_strtNendTmstmp = int(k2[1]) - int(k2[0])
					assert dl_strtNendTmstmp >= 0, print(k2)
					if dl_strtNendTmstmp > self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats:
						continue

				if self.egoVlpV2_vis2textSim_labler:
					if len(v2['scores']) != 1:
						continue

				if self.egoVlpV2_vis2textSim_labler:
					self.lst_dtpnts.append({'take_name': k1,
												'startNend_clipName': v2['startNend_clipName'],
												'startNend_frameIdx': v2['startNend_frameIdx'],
												'startNend_timestamp': k2,
												}
											)
				else:
					self.lst_dtpnts.append({'take_name': k1,
											'startNend_clipName': v2['startNend_clipName'],
											'startNend_frameIdx': v2['startNend_frameIdx'],
											'startNend_timestamp': k2,
											'timestamp': v2['timestamp']})

				if  self.isLemma_dataset:
					assert k1 in self.lemmaDataset_dct
					assert (k2[1], k2[2]) in self.lemmaDataset_dct[k1], print(k2, (k2[1], k2[2]), list(self.lemmaDataset_dct[k1].keys())[:2])
					self.lst_dtpnts[-1]["list_egoNexoSuffixes"] = self.lemmaDataset_dct[k1][(k2[1], k2[2])]["list_egoNexoSuffixes"]
				else:
					if len(k2) == 2:
						if ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[0]}_{k2[1]}.mp4"):
							pass
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[0])}_{k2[1]}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (int(k2[0]), k2[1])
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[0]}_{int(k2[1])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], int(k2[1]))
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[0])}_{int(k2[1])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (int(k2[0]), int(k2[1]))
						else:
							raise ValueError
					else:
						if ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[1]}_{k2[2]}.mp4"):
							pass
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[1])}_{k2[2]}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], int(k2[1]), k2[2])
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{k2[1]}_{int(k2[2])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], k2[1], int(k2[2]))
						elif ospif(f"{self.datapoint_videoClips_dir}/aria/{k1}/"+\
									f"{v2['startNend_clipName'][0]}_{v2['startNend_clipName'][1]}__{v2['startNend_frameIdx'][0]}_{v2['startNend_frameIdx'][1]}__{int(k2[1])}_{int(k2[2])}.mp4"):
							self.lst_dtpnts[-1]['startNend_timestamp'] = (k2[0], int(k2[1]), int(k2[2]))
						else:
							raise ValueError

				if self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
					if self.is_multiPseudolabler:
						raise ValueError
					self.lst_dtpnts[-1]['best_exo_views'] = v2['best_exo_views']
				else:
					if self.is_multiPseudolabler:
						self.lst_dtpnts[-1]['scores'] = []
						for ele__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs in\
								lst__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs:
							self.lst_dtpnts[-1]['scores'].append(
										ele__tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs[k1][k2]['scores']
									)
							if self.all_views == ['1', '2', '3', '4', ]:
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][1:]
							elif self.all_views in [['aria', '1'], ['aria', '2'], ['aria', '3'], ['aria', '4']]:
								exo_strtIdx = int(self.all_views[-1])
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][:1] +\
																	self.lst_dtpnts[-1]['scores'][-1][exo_strtIdx: (exo_strtIdx + 1)]
					else:
						if self.egoVlpV2_vis2textSim_labler:
							if self.all_views == ['1', '2', '3', '4', ]:
								raise NotImplementedError

							self.lst_dtpnts[-1]['scores'] = v2['scores'][0]
							self.lst_dtpnts[-1]['scores_captioner'] = tkNm_2_strtNendTmstmp_cptnrScrs[k1][(round(k2[0], 4), round(k2[1], 4), round(k2[2], 4) )]['scores']
						else:
							self.lst_dtpnts[-1]['scores'] = v2['scores']
							if self.all_views == ['1', '2', '3', '4', ]:
								self.lst_dtpnts[-1]['scores'] = self.lst_dtpnts[-1]['scores'][1:]
							elif self.all_views in [['aria', '1'], ['aria', '2'], ['aria', '3'], ['aria', '4']]:
								exo_strtIdx = int(self.all_views[-1])
								self.lst_dtpnts[-1]['scores'][-1] = self.lst_dtpnts[-1]['scores'][-1][:1] +\
																	self.lst_dtpnts[-1]['scores'][-1][exo_strtIdx: (exo_strtIdx + 1)]

		self.egoVlpV2_takeVideoFeats_takeName2camId2featName = None
		if self.use_egoVlpV2_takeVideoFeats:
			assert ospid(self.egoVlpV2_takeVideoFeats_dir)

			assert ospif(self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp)
			self.egoVlpV2_takeVideoFeats_takeName2camId2featName = pkl_ld(self.egoVlpV2_takeVideoFeats_takeName2camId2featName_fp)

		self.total_num_samples = min(self.total_num_samples, len(self.lst_dtpnts))

		if self.task_type != "classify_oneHot_bestExoPred":
			self.class_weights = compute_classWeights(len(self.all_views),
													  self.lst_dtpnts,
													  self.is_multiPseudolabler,
													  self.topK_multiPseudolabler,
													  self.bordaCount_multiPseudolabler,
												  self.multiBestViewAggregator_multiPseudoLabler)
		else:
			self.class_weights = np.array([0, 0, 0, 0])

	def __len__(self):
		return self.total_num_samples	# 2, 6, 24, self.total_num_samples

	def __getitem__(self, index):
		dtpnt_idx = index
		dtpnt = self.lst_dtpnts[dtpnt_idx]
		al_frms = []
		al_cameraPoses = None
		if self.use_relativeCameraPoseLoss:
			al_cameraPoses = []
		for vw in self.all_views:
			tk_nm = dtpnt['take_name']
			if len(dtpnt['startNend_timestamp']) == 3:
				cntr_tmstmp = dtpnt['startNend_timestamp'][0]
				strt_tmstmp = dtpnt['startNend_timestamp'][1]
				end_tmstmp = dtpnt['startNend_timestamp'][2]
			else:
				cntr_tmstmp = dtpnt['timestamp'] if ('timestamp' in dtpnt) else None
				strt_tmstmp = dtpnt['startNend_timestamp'][0]
				end_tmstmp = dtpnt['startNend_timestamp'][1]
			if self.use_egoVlpV2_takeVideoFeats:
				if self.use_relativeCameraPoseLoss:
					raise NotImplementedError

				assert tk_nm in self.egoVlpV2_takeVideoFeats_takeName2camId2featName
				egoVlpV2_takeVideoFeats_camId2featName = self.egoVlpV2_takeVideoFeats_takeName2camId2featName[tk_nm]

				assert vw in egoVlpV2_takeVideoFeats_camId2featName
				ft_nm = egoVlpV2_takeVideoFeats_camId2featName[vw]

				ft_fp = f"{self.egoVlpV2_takeVideoFeats_dir}/{ft_nm}"
				ft = torch.load(ft_fp, map_location="cpu")

				srt_tmstmp_int = int(strt_tmstmp)
				end_tmstmp_int = int(end_tmstmp)
				assert 0 <= end_tmstmp_int - srt_tmstmp_int <= self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats

				if self.use_egoVlpV2_takeVideoFeats_usingCenterTime:
					srt_tmstmp_int = end_tmstmp_int = int(cntr_tmstmp)
				
				frms = None
				ft_slc = None
				tmstmp_cnt = 0
				for tmstmp_int in range(srt_tmstmp_int, end_tmstmp_int + 1):
					if tmstmp_int < len(ft):
						ft_slc = ft[tmstmp_int]
						if frms is None:
							frms = ft_slc
						else:
							frms = torch.cat((frms, ft_slc))
					else:
						break
					tmstmp_cnt += 1

				if frms is None:
					assert ft_slc is None
					ft_slc = torch.zeros(4096)
					frms = torch.zeros(0)
				else:
					assert ft_slc is not None

				if self.use_egoVlpV2_takeVideoFeats_usingCenterTime:
					frms = ft_slc
					assert len(frms) == 4096
				elif self.use_egoVlpV2_takeVideoFeats_usingStartNendTime:
					if self.padFeatWithZero_use_egoVlpV2_takeVideoFeats_usingStartNendTime:
						frms = torch.cat((frms, 
											torch.zeros(4096 * (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1 - tmstmp_cnt))))
					else:
						frms = torch.cat([frms] +\
											([ft_slc] * (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1 - tmstmp_cnt)))
					assert len(frms) == (self.maxStartNendTimeDiff_use_egoVlpV2_takeVideoFeats + 1) * 4096
			elif self.use_videoLlama_feats:
				if self.use_relativeCameraPoseLoss:
					raise NotImplementedError

				strt_clpNm = dtpnt['startNend_clipName'][0]
				end_clpNm = dtpnt['startNend_clipName'][1]
				strt_frmIdx = dtpnt['startNend_frameIdx'][0]
				end_frmIdx = dtpnt['startNend_frameIdx'][1]

				ft_fp = f"{self.videoLlama_feats_dir}/{vw}/{tk_nm}/{strt_clpNm}_{end_clpNm}__{strt_frmIdx}_{end_frmIdx}.pt"
				assert ospif(ft_fp), print(ft_fp)

				ft = torch.load(ft_fp, map_location="cpu")
				if self.videoLlama_feats_seqAggregation == "mean":
					ft = torch.mean(ft, dim=0).unsqueeze(0)
				frms = ft.reshape((ft.shape[0] * ft.shape[1]))
			else:
				if self.use_datapointVideoClips:
					if self.isLemma_dataset:
						lst_egoNexoSffxs = dtpnt['list_egoNexoSuffixes']
						lst_imgSffxs = []
						for egoNexo_sffx in lst_egoNexoSffxs:
							if vw == "fpv1":
								lst_imgSffxs.append(egoNexo_sffx[0])
							elif vw == "master":
								lst_imgSffxs.append(egoNexo_sffx[1])
							else:
								raise ValueError

						assert self.frame_height == self.frame_width

						frms = []
						for img_sffx in lst_imgSffxs:
							img_fp = f"{self.datapoint_videoClips_dir}/{img_sffx}"
							assert os.path.isfile(img_fp)
							tmp_img = np.array(Image.open(img_fp).resize((self.frame_height, self.frame_width), resample=Image.BICUBIC))
							assert tmp_img.dtype == np.uint8
							frms.append(torch.from_numpy(tmp_img))

						frms = torch.stack(frms).float() / 255
					else:
						strt_clpNm = dtpnt['startNend_clipName'][0]
						end_clpNm = dtpnt['startNend_clipName'][1]
						strt_frmIdx = dtpnt['startNend_frameIdx'][0]
						end_frmIdx = dtpnt['startNend_frameIdx'][1]

						clp_pth = f"{self.datapoint_videoClips_dir}/{vw}/{tk_nm}/"+\
									f"{strt_clpNm}_{end_clpNm}__{strt_frmIdx}_{end_frmIdx}__{strt_tmstmp}_{end_tmstmp}.mp4"

						frms, frm_idxs = load_datapointVideo_egoExoNarrate(clp_pth,
																n_frms=self.num_frames,
																height=self.frame_height,
																width=self.frame_width,
																dont_square_frames=self.dont_square_frames,)

					if self.use_relativeCameraPoseLoss:
						if tk_nm in self.tkNm2cameraPose:
							has_cameraPose = True
							cameraPose_thisTake = self.tkNm2cameraPose[tk_nm]
						else:
							has_cameraPose = False
							cameraPose_thisTake = self.tkNm2cameraPose[list(self.tkNm2cameraPose.keys())[0]]

						relativeCameraPoseLoss_global_frameIdxs = []

						if self.relativeCameraPoseLoss_frameType == "center":
							if has_cameraPose:
								relativeCameraPoseLoss_frameIdx = frm_idxs[len(frm_idxs) // 2]
								relativeCameraPoseLoss_global_frameIdxs = [(int(strt_clpNm) * 900) + int(strt_frmIdx) + int(relativeCameraPoseLoss_frameIdx)]
							else:
								relativeCameraPoseLoss_global_frameIdxs = [0]
						elif self.relativeCameraPoseLoss_frameType == "all":
							if has_cameraPose:						
								for relativeCameraPoseLoss_frameIdx in frm_idxs:
									relativeCameraPoseLoss_global_frameIdx = (int(strt_clpNm) * 900) + int(strt_frmIdx) + int(relativeCameraPoseLoss_frameIdx)
									relativeCameraPoseLoss_global_frameIdxs.append(relativeCameraPoseLoss_global_frameIdx)
							else:
								relativeCameraPoseLoss_global_frameIdxs = [0] * len(frm_idxs)
						else:
							raise NotImplementedError

						cameraPoses_thisVw = []
						if vw == 'aria':
							assert 'ego' in cameraPose_thisTake

							for relativeCameraPoseLoss_global_frameIdx in relativeCameraPoseLoss_global_frameIdxs:
								assert str(relativeCameraPoseLoss_global_frameIdx) in cameraPose_thisTake['ego']
								cameraPose_thisVw = cameraPose_thisTake['ego'][str(relativeCameraPoseLoss_global_frameIdx)]
								cameraPoses_thisVw.append(cameraPose_thisVw)

						else:
							assert vw in cameraPose_thisTake

							for relativeCameraPoseLoss_global_frameIdx in relativeCameraPoseLoss_global_frameIdxs:
								cameraPose_thisVw = cameraPose_thisTake[vw]
								cameraPoses_thisVw.append(cameraPose_thisVw)

						al_cameraPoses.append(cameraPoses_thisVw)

					frms = frms.permute(3, 0, 1, 2) # (T, H, W, C -> C, T, H, W)
					assert self.transforms is not None
					frms = self.transforms(frms)
					frms = frms.permute(1, 2, 3, 0)
				else:
					if self.use_relativeCameraPoseLoss:
						raise NotImplementedError

					clp_dr = f"{self.videoClips_dir}/{vw}/{tk_nm}"
					assert os.path.isdir(clp_dr)
					frms = load_video_egoExoNarrate(clp_dr,
													dtpnt['startNend_clipName'],
													dtpnt['startNend_frameIdx'],
													n_frms=self.num_frames,
													height=self.frame_height,
													width=self.frame_width)
					frms = frame_normalize(frms, input_frame_norm_type=self.recog_arc)

			al_frms.append(frms)

		ref_camerPoses = None
		if self.use_relativeCameraPoseLoss:
			ref_camerPoses = al_cameraPoses[0]

		al_frms = torch.stack(al_frms)

		al_rel_cameraPoses = None
		if self.use_relativeCameraPoseLoss:
			al_rel_cameraPoses = []
			for camera_poses in al_cameraPoses:

				if self.relativeCameraPoseLoss_refType == "first_view":
					rel_cameraPoses_thisVw = []
					for camerPose_idx, camera_pose in enumerate(camera_poses):
						rel_cameraPose = get_rel_ce(ref_camerPoses[camerPose_idx],
													camera_pose,
													return_angles=self.relativeCameraPoseLoss_rotationInAngles,
													return_quarts=self.relativeCameraPoseLoss_rotationInQuarts,
													return_onlyRotation=self.relativeCameraPoseLoss_rotationOnly,
													return_coord_angles=self.relativeCameraPoseLoss_coordsInAngles,
													return_coord_normalized=self.relativeCameraPoseLoss_coordsNormalized,)
						rel_cameraPoses_thisVw.append(rel_cameraPose)

					al_rel_cameraPoses.append(rel_cameraPoses_thisVw)
				elif self.relativeCameraPoseLoss_refType == "all_views":
					for camera_poses2 in al_cameraPoses:
						rel_cameraPoses_thisVwPr = []
						for camerPose_idx, camera_pose in enumerate(camera_poses2):
							rel_cameraPose = get_rel_ce(camera_poses[camerPose_idx], camera_pose,
														return_angles=self.relativeCameraPoseLoss_rotationInAngles,
														return_quarts=self.relativeCameraPoseLoss_rotationInQuarts,
														return_onlyRotation=self.relativeCameraPoseLoss_rotationOnly,
														return_coord_angles=self.relativeCameraPoseLoss_coordsInAngles,
														return_coord_normalized=self.relativeCameraPoseLoss_coordsNormalized,)
							rel_cameraPoses_thisVwPr.append(rel_cameraPose)

						al_rel_cameraPoses.append(rel_cameraPoses_thisVwPr)

			al_rel_cameraPoses = torch.from_numpy(np.array(al_rel_cameraPoses)).float() # np.concatenate -> np.array

		class_weights = torch.from_numpy(self.class_weights).float()

		if ('best_exo_views' in dtpnt) and (isinstance(dtpnt['best_exo_views'][0], str)):
			dtpnt_scrs_tmp = [0, 0, 0, 0]
			dtpnt_scrs_tmp[int(dtpnt['best_exo_views'][0]) - 1] = 1
			dtpnt_scrs_tnsr = torch.tensor(dtpnt_scrs_tmp).float()
		else:
			dtpnt_scrs_tnsr = torch.tensor(dtpnt['scores']).float()
		if len(dtpnt_scrs_tnsr.shape) == 1:
			dtpnt_scrs_tnsr = dtpnt_scrs_tnsr.unsqueeze(0)
		if self.egoVlpV2_vis2textSim_labler:
			dtpnt_cptnr_scrs_tnsr = torch.tensor(dtpnt['scores_captioner']).float()
			if len(dtpnt_cptnr_scrs_tnsr.shape) == 1:
				dtpnt_cptnr_scrs_tnsr = dtpnt_cptnr_scrs_tnsr.unsqueeze(0)

		if self.task_type in ["classify_oneHot", "match_dist"]:
			dtpnt_scrs = dtpnt['scores']
			if self.task_type == "classify_oneHot":
				if self.is_multiPseudolabler:
					if self.multiBestViewAggregator_multiPseudoLabler:
						assert len(dtpnt_scrs) == 3
						bstIdx2cnt_thisEle = {}
						best_idxs_all = []
						for dtpnt_scrs_ele in dtpnt_scrs:
							best_idxs = []
							for bstVw_idx in np.where(dtpnt_scrs_ele == np.max(dtpnt_scrs_ele))[0].tolist():
								best_idxs.append(bstVw_idx)	
								if bstVw_idx not in bstIdx2cnt_thisEle:
									bstIdx2cnt_thisEle[bstVw_idx] = 0
								bstIdx2cnt_thisEle[bstVw_idx] += 1
							best_idxs_all.append(best_idxs)
						cnt2bstIdxs_thisEle = {}
						for bst_idx, cnt in bstIdx2cnt_thisEle.items():
							if cnt not in cnt2bstIdxs_thisEle:
								cnt2bstIdxs_thisEle[cnt] = set()
							cnt2bstIdxs_thisEle[cnt].add(bst_idx)
						cntNbstIdxs_thisEle = []
						for cnt, bst_idxs in cnt2bstIdxs_thisEle.items():
							cntNbstIdxs_thisEle.append((cnt, bst_idxs))
						srtd_cntNbstIdxs_thisEle = sorted(cntNbstIdxs_thisEle)[::-1]
						if srtd_cntNbstIdxs_thisEle[0][0] in [1]:
							best_idxs = best_idxs_all[0]
						else:
							best_idxs = list(srtd_cntNbstIdxs_thisEle[0][1])
					else:
						dtpnt_scrs_argsrt = np.argsort(dtpnt_scrs, axis=1)[:, ::-1]
						if self.bordaCount_multiPseudolabler:
							vwIdx2bordaCount = {}
							vwIdx2numVotes = {}
							for colIdx_dtpnt_scrs_argsrt in range(dtpnt_scrs_argsrt.shape[1]):
								col_dtpnt_scrs_argsrt = dtpnt_scrs_argsrt[:, colIdx_dtpnt_scrs_argsrt]
								col_bordaCount = dtpnt_scrs_argsrt.shape[1] - 1 - colIdx_dtpnt_scrs_argsrt
								for vw_idx in col_dtpnt_scrs_argsrt:
									if vw_idx not in vwIdx2bordaCount:
										vwIdx2bordaCount[vw_idx] = 0
										assert vw_idx not in vwIdx2numVotes
										vwIdx2numVotes[vw_idx] = 0

									vwIdx2bordaCount[vw_idx] += col_bordaCount
									vwIdx2numVotes[vw_idx] += 1

							lst_bordaCountNvwIdx = list()
							for vw_idx, total_bordaCount in vwIdx2bordaCount.items():
								assert  vwIdx2numVotes[vw_idx] == len(dtpnt_scrs_argsrt)
								lst_bordaCountNvwIdx.append((total_bordaCount, vw_idx))
							srtdLst_bordaCountNvwIdx = sorted(lst_bordaCountNvwIdx)[::-1]

							best_idxs = []
							for ele_srtdLst_bordaCountNvwIdx in srtdLst_bordaCountNvwIdx:
								if ele_srtdLst_bordaCountNvwIdx[0] == srtdLst_bordaCountNvwIdx[0][0]:
									best_idxs.append(ele_srtdLst_bordaCountNvwIdx[1])
								else:
									break
						else:
							dtpnt_scrs_argsrt_topK = dtpnt_scrs_argsrt[:, :self.topK_multiPseudolabler]	

							""" considers vote count and average rank"""
							idx2numVotesNtotalWeight = {}
							for row_dtpnt_scrs_argsrt_topK in dtpnt_scrs_argsrt_topK:
								for ele_idx, ele in enumerate(row_dtpnt_scrs_argsrt_topK):
									if ele not in idx2numVotesNtotalWeight:
										idx2numVotesNtotalWeight[ele] = [0, 0]

									idx2numVotesNtotalWeight[ele][0] += 1

									ele_weight =  self.topK_multiPseudolabler - ele_idx
									idx2numVotesNtotalWeight[ele][1] += ele_weight

							numVotes2avgWeightsNidxs = {}
							for vw_idx, num_votesNtotalWeight in idx2numVotesNtotalWeight.items():
								if num_votesNtotalWeight[0] not in numVotes2avgWeightsNidxs:
									numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]] = set()
								numVotes2avgWeightsNidxs[num_votesNtotalWeight[0]].add((num_votesNtotalWeight[1] / max(num_votesNtotalWeight[0], 1), vw_idx ))
							lst_numVotesNavgWeightsNidxs = []
							for num_votes, avgWeightsNvwIdxs in numVotes2avgWeightsNidxs.items():
								lst_numVotesNavgWeightsNidxs.append((num_votes, list(avgWeightsNvwIdxs)))
							srtdLst_numVotesNavgWeightsNidxs = sorted(lst_numVotesNavgWeightsNidxs)[::-1]
							avgWeightsNidxs_wHighestVotes = srtdLst_numVotesNavgWeightsNidxs[0][1]

							srtdLst_avgWeightsNidxs_wHighestVotes = sorted(avgWeightsNidxs_wHighestVotes)[::-1]
							best_idxs = []
							for ele_srtdLst_avgWeightsNidxs_wHighestVotes in srtdLst_avgWeightsNidxs_wHighestVotes:
								if ele_srtdLst_avgWeightsNidxs_wHighestVotes[0] == srtdLst_avgWeightsNidxs_wHighestVotes[0][0]:
									best_idxs.append(ele_srtdLst_avgWeightsNidxs_wHighestVotes[1])
					lbl_idx = best_idxs[0]
				else:
					lbl_idx = np.argmax(dtpnt_scrs)

				lbl = torch.tensor([lbl_idx]).long()
			elif self.task_type == "match_dist":
				if self.is_multiPseudolabler:
					print("need to normalize scores")
					raise NotImplementedError
					dtpnt_scrs = np.mean(dtpnt_scrs, axis=0)
				lbl = torch.tensor(dtpnt_scrs).float()

			lbl_multiHot = torch.zeros((len(self.all_views))).float()
			if (self.task_type in ["classify_oneHot"]) and (self.is_multiPseudolabler):
				for bstVw_idx in best_idxs:
					lbl_multiHot[bstVw_idx] = 1	
			else:
				for bstVw_idx in np.where(dtpnt_scrs == np.max(dtpnt_scrs))[0].tolist():
					lbl_multiHot[bstVw_idx] = 1	

		elif self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
			lbl = torch.zeros((len(self.all_views))).float()
			for bst_ex_vw in dtpnt['best_exo_views']:
				lbl[int(bst_ex_vw) - 1] = 1	

			if self.task_type == "classify_oneHot_bestExoPred":
				lbl_multiHot = lbl
				lbl = torch.tensor([int(dtpnt['best_exo_views'][0])]).long() - 1	
		else:
			raise NotImplementedError

		if self.task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
			if self.use_relativeCameraPoseLoss:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
				else:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
			else:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights
				else:
					return al_frms, lbl, lbl_multiHot, dtpnt_scrs_tnsr, class_weights
		else:
			if self.use_relativeCameraPoseLoss:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
				else:
					return al_frms, lbl, dtpnt_scrs_tnsr, class_weights, al_rel_cameraPoses, has_cameraPose
			else:
				if self.egoVlpV2_vis2textSim_labler:
					return al_frms, lbl, dtpnt_scrs_tnsr, dtpnt_cptnr_scrs_tnsr, class_weights
				else:
					return al_frms, lbl, dtpnt_scrs_tnsr, class_weights


class test_dataset(object):
	def __init__(self, args, **kwargs):
		self.lemmaDataset_dct =\
			pkl_ld("data/lemma/v1/misc/"+\
				   "take__2__startNendEgoImageSuffix__2__timestamp_n_startNendClipName_n_startNendFrameIdx_n_listAtomicDescriptions_n_listImageSuffixes__val.pkl")

		self.args = args
		self.kwargs = kwargs
		self.batch_size = kwargs["batch_size"]
		self.all_views = kwargs['all_views']
		self.num_frames = kwargs['num_frames']
		self.frame_height = kwargs['frame_height']
		self.frame_width = kwargs['frame_width']
		self.dont_square_frames = kwargs["dont_square_frames"] if ("dont_square_frames" in kwargs) else False
		self.videoClips_dir = kwargs["videoClips_dir"] if ("videoClips_dir" in kwargs) else None
		self.datapoint_videoClips_dir = kwargs["datapoint_videoClips_dir"] if ("datapoint_videoClips_dir" in kwargs) else None
		datapoints_filePath = kwargs["testDatapoints_filePath"]
		self.recog_arc = kwargs['recog_arc']
		self.task_type = kwargs['task_type']
		self.use_datapointVideoClips = kwargs["use_datapointVideoClips"] if ("use_datapointVideoClips" in kwargs) else False

		self.isLemma_dataset = kwargs["isLemma_dataset"] if ("isLemma_dataset" in kwargs) else False

		self.transforms = None
		if self.use_datapointVideoClips:
			assert self.frame_height == self.frame_width

			frame_mean, frame_std = frame_normalize(None, 
													input_frame_norm_type=self.recog_arc,
													return_meanNstd=True)

			frame_normalize_ = NormalizeVideo(mean=frame_mean, std=frame_std)

			self.transforms = transforms.Compose([
			    transforms.Resize(self.frame_height),
			    transforms.CenterCrop(self.frame_height),
			    frame_normalize_,
			])

		assert os.path.isfile(datapoints_filePath), print(datapoints_filePath)
		with open(datapoints_filePath, "rb") as fi:
			self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs = pickle.load(fi)
		self.lst_dtpnts = []
		self.total_num_samples = 0
		for k1, v1 in self.tkNm_2_strtNendTmstmp_2_tmstmpNstrtNendClpnmNstrtNendFrmIdxNscrs.items():
			self.total_num_samples += len(v1)
			for k2, v2 in v1.items():
				self.lst_dtpnts.append({'take_name': k1,
										'startNend_clipName': v2['startNend_clipName'],
										'startNend_frameIdx': v2['startNend_frameIdx'],
										'startNend_timestamp': k2,
										'timestamp': v2['timestamp']})

				if self.isLemma_dataset:
					assert k1 in self.lemmaDataset_dct
					assert (k2[1], k2[2]) in self.lemmaDataset_dct[k1], print(k2, (k2[1], k2[2]), list(self.lemmaDataset_dct[k1].keys())[:2])
					self.lst_dtpnts[-1]["list_egoNexoSuffixes"] = self.lemmaDataset_dct[k1][(k2[1], k2[2])]["list_egoNexoSuffixes"]

				if self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
					self.lst_dtpnts[-1]['best_exo_views'] = v2['best_exo_views']
				else:
					self.lst_dtpnts[-1]['scores'] = v2['scores']
					if self.all_views == ['1', '2', '3', '4', ]:
						self.lst_dtpnts[-1]['scores'] = self.lst_dtpnts[-1]['scores'][1:]
					elif self.all_views in [['aria', '1'], ['aria', '2'], ['aria', '3'], ['aria', '4']]:
						exo_strtIdx = int(self.all_views[-1])
						self.lst_dtpnts[-1]['scores'] = self.lst_dtpnts[-1]['scores'][:1] +\
															self.lst_dtpnts[-1]['scores'][exo_strtIdx: (exo_strtIdx + 1)]

	def __len__(self):
		return self.total_num_samples   # 2, 6, self.total_num_samples

	def __getitem__(self, index):
		dtpnt_idx = index
		dtpnt = self.lst_dtpnts[dtpnt_idx]
		al_frms = []
		for vw in self.all_views:
			tk_nm = dtpnt['take_name']
			if len(dtpnt['startNend_timestamp']) == 3:
				cntr_tmstmp = dtpnt['startNend_timestamp'][0]
				strt_tmstmp = dtpnt['startNend_timestamp'][1]
				end_tmstmp = dtpnt['startNend_timestamp'][2]
			else:
				cntr_tmstmp = dtpnt['timestamp']
				strt_tmstmp = dtpnt['startNend_timestamp'][0]
				end_tmstmp = dtpnt['startNend_timestamp'][1]

			if self.use_datapointVideoClips:
				if self.isLemma_dataset:
					lst_egoNexoSffxs = dtpnt['list_egoNexoSuffixes']
					lst_imgSffxs = []
					for egoNexo_sffx in lst_egoNexoSffxs:
						if vw == "fpv1":
							lst_imgSffxs.append(egoNexo_sffx[0])
						elif vw == "master":
							lst_imgSffxs.append(egoNexo_sffx[1])
						else:
							raise ValueError

					assert self.frame_height == self.frame_width

					frms = []
					for img_sffx in lst_imgSffxs:
						img_fp = f"{self.datapoint_videoClips_dir}/{img_sffx}"
						assert os.path.isfile(img_fp), print(img_fp)
						tmp_img = np.array(Image.open(img_fp).resize((self.frame_height, self.frame_width), resample=Image.BICUBIC))
						assert tmp_img.dtype == np.uint8
						frms.append(torch.from_numpy(tmp_img))

					frms = torch.stack(frms).float() / 255
				else:
					strt_clpNm = dtpnt['startNend_clipName'][0]
					end_clpNm = dtpnt['startNend_clipName'][1]
					strt_frmIdx = dtpnt['startNend_frameIdx'][0]
					end_frmIdx = dtpnt['startNend_frameIdx'][1]

					clp_pth = f"{self.datapoint_videoClips_dir}/{vw}/{tk_nm}/"+\
								f"{strt_clpNm}_{end_clpNm}__{strt_frmIdx}_{end_frmIdx}__{strt_tmstmp}_{end_tmstmp}.mp4"

					frms, frm_idxs = load_datapointVideo_egoExoNarrate(clp_pth,
															n_frms=self.num_frames,
															height=self.frame_height,
															width=self.frame_width,
															dont_square_frames=self.dont_square_frames,)

				frms = frms.permute(3, 0, 1, 2) # (T, H, W, C -> C, T, H, W)
				assert self.transforms is not None
				frms = self.transforms(frms)
				frms = frms.permute(1, 2, 3, 0)
			else:
				clp_dr = f"{self.videoClips_dir}/{vw}/{dtpnt['take_name']}"
				assert os.path.isdir(clp_dr)
				frms = load_video_egoExoNarrate(clp_dr,
												dtpnt['startNend_clipName'],
												dtpnt['startNend_frameIdx'],
												n_frms=self.num_frames,
												height=self.frame_height,
												width=self.frame_width)
				frms = frame_normalize(frms, input_frame_norm_type=self.recog_arc)
			al_frms.append(frms)
		al_frms = torch.stack(al_frms)

		if self.task_type == "classify_oneHot":
			lbl = torch.tensor([np.argmax(dtpnt['scores'])]).long()
		elif self.task_type == "match_dist":
			lbl = torch.tensor(dtpnt['scores']).float()
		elif self.task_type in ["classify_oneHot_bestExoPred", "classify_multiHot_bestExoPred"]:
			lbl = torch.zeros((len(self.all_views))).float()
			for bst_ex_vw in dtpnt['best_exo_views']:
				lbl[int(bst_ex_vw) - 1] = 1	

			if self.task_type == "classify_oneHot_bestExoPred":
				lbl_multiHot = lbl
				lbl = torch.tensor([int(dtpnt['best_exo_views'][0])]).long() - 1	
		else:
			raise NotImplementedError

		if self.task_type == "classify_oneHot_bestExoPred":
			return al_frms, lbl, index, lbl_multiHot
		else:
			return al_frms, lbl, index
