from models import pol
from common.utils import *
from common.dist_utils import *
from common.logger import *
from datasets.utils import *

import os
import numpy as np
from tqdm import tqdm

import cv2 as cv

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_n_val(train_loader,
				val_loader,
				writer,
				args,
				**kwargs):
	run_dir = kwargs["run_dir"]

	ckpt_dir = os.path.join(run_dir, "data")
	if (not os.path.isdir(ckpt_dir)) and is_main_process(args):
		os.makedirs(ckpt_dir)

	load_ckpt = False
	if os.path.isfile(os.path.join(run_dir, "data/valLastCkpt.pth")):
		load_ckpt = True

		print("Loading last checkpoint!")
		loaded_ckpt = torch.load(os.path.join(run_dir,
											  "data/valLastCkpt.pth"),
								 map_location="cpu")

		kwargs = loaded_ckpt["args"]	

	num_epochs = kwargs["epochs"]
	num_samples = kwargs["num_trainSamples"]
	num_valSamples = kwargs["num_valSamples"]
	batch_size = kwargs["batch_size"]

	optimizer_type = kwargs["optimizer_type"] if ("optimizer_type" in kwargs) else "adam_w"
	assert optimizer_type in ["adam_w", "adam"]

	frame_height = kwargs["frame_height"]
	frame_width = kwargs["frame_width"]

	frame_colorJitter = kwargs["frame_colorJitter"] if ("frame_colorJitter" in kwargs) else [0, 0, 0]
	train_transforms_colorJitter = None
	train_transforms_normalize = None
	if (frame_colorJitter != [0, 0, 0]) and (frame_colorJitter != [0, 0]):
		if hasattr(train_loader, "transforms_colorJitter"):
			train_transforms_colorJitter = train_loader.transforms_colorJitter
			train_transforms_normalize = train_loader.transforms_normalize
		else:
			train_transforms_colorJitter = train_loader.dataset.transforms_colorJitter
			train_transforms_normalize = train_loader.dataset.transforms_normalize

	if kwargs["distributed"]:
		num_trainIters = kwargs["num_trainIterations"]
		num_valIters = kwargs["num_valIterations"]
		device = torch.device(kwargs["device"])
	else:
		device = (
			torch.device("cuda", 0)
			if torch.cuda.is_available()
			else torch.device("cpu")
		)
		n_available_gpus = torch.cuda.device_count()

	task_type = kwargs['task_type']
	recog_arc = kwargs["recog_arc"]
	unfreeze_videoEncoder = kwargs["unfreeze_videoEncoder"] if ("unfreeze_videoEncoder" in kwargs) else False

	assert task_type in ['classify_oneHot', 'match_dist', 'classify_oneHot_bestExoPred', 'classify_multiHot_bestExoPred']
	task_isBestExoPred = task_type in ['classify_oneHot_bestExoPred', 'classify_multiHot_bestExoPred']

	use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
														if ("use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
															False
	use_egoVlpV2_takeVideoFeats_usingCenterTime = kwargs["use_egoVlpV2_takeVideoFeats_usingCenterTime"]\
														if ("use_egoVlpV2_takeVideoFeats_usingCenterTime" in kwargs) else\
															False
	use_egoVlpV2_takeVideoFeats = use_egoVlpV2_takeVideoFeats_usingStartNendTime or\
									use_egoVlpV2_takeVideoFeats_usingCenterTime	

	use_videoLlama_feats = kwargs["use_videoLlama_feats"] if ("use_videoLlama_feats" in kwargs) else False

	use_preExtractedFeats = use_egoVlpV2_takeVideoFeats or\
								use_videoLlama_feats 

	egoVlpV2_vis2textSim_labler = kwargs["egoVlpV2_vis2textSim_labler"] if ("egoVlpV2_vis2textSim_labler" in kwargs) else False

	use_minMultiHotLoss	= kwargs["use_minMultiHotLoss"] if ("use_minMultiHotLoss" in kwargs) else False
	use_randMultiHotLoss	= kwargs["use_randMultiHotLoss"] if ("use_randMultiHotLoss" in kwargs) else False
	use_bceMultiHotLoss	= kwargs["use_bceMultiHotLoss"] if ("use_bceMultiHotLoss" in kwargs) else False				
	use_klLoss = kwargs["use_klLoss"] if ("use_klLoss" in kwargs) else False	

	balanceCLasses_inLoss = kwargs["balanceCLasses_inLoss"] if ("balanceCLasses_inLoss" in kwargs) else False

	if use_bceMultiHotLoss:
		if balanceCLasses_inLoss:
			raise NotImplementedError

	use_relativeCameraPoseLoss = kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False
	relativeCameraPoseLoss_coordsInAngles = kwargs["relativeCameraPoseLoss_coordsInAngles"] if ("relativeCameraPoseLoss_coordsInAngles" in kwargs) else False
	relativeCameraPoseLoss_coordsAsClasses = kwargs["relativeCameraPoseLoss_coordsAsClasses"] if ("relativeCameraPoseLoss_coordsAsClasses" in kwargs) else False
	relativeCameraPoseLoss_coordsClassSize = kwargs["relativeCameraPoseLoss_coordsClassSize"] if ("relativeCameraPoseLoss_coordsClassSize" in kwargs) else 10
	relativeCameraPoseLoss_rotationOnly = kwargs["relativeCameraPoseLoss_rotationOnly"] if ("relativeCameraPoseLoss_rotationOnly" in kwargs) else False
	relativeCameraPoseLoss_rotationInAngles = kwargs["relativeCameraPoseLoss_rotationInAngles"] if ("relativeCameraPoseLoss_rotationInAngles" in kwargs) else False
	relativeCameraPoseLoss_rotationInQuarts = kwargs["relativeCameraPoseLoss_rotationInQuarts"] if ("relativeCameraPoseLoss_rotationInQuarts" in kwargs) else False
	relativeCameraPoseLoss_rotationAsClasses = kwargs["relativeCameraPoseLoss_rotationAsClasses"] if ("relativeCameraPoseLoss_rotationAsClasses" in kwargs) else False
	relativeCameraPoseLoss_rotationClassSize = kwargs["relativeCameraPoseLoss_rotationClassSize"] if ("relativeCameraPoseLoss_rotationClassSize" in kwargs) else 10
	maskOut_invalidRelativeCameraPoseLoss_inTraining = kwargs["maskOut_invalidRelativeCameraPoseLoss_inTraining"] if ("maskOut_invalidRelativeCameraPoseLoss_inTraining" in kwargs) else False
	relativeCameraPoseLoss_lossType = kwargs["relativeCameraPoseLoss_lossType"] if ("relativeCameraPoseLoss_lossType" in kwargs) else 'l2'	
	# Accept both CLI spellings for loss weight (train.py uses 'relativeCameraPoseLoss-weight')


	if "relativeCameraPoseLoss_weight" in kwargs:
		relativeCameraPoseLoss_lossWeight = kwargs["relativeCameraPoseLoss_weight"]
	elif "relativeCameraPoseLoss_lossWeight" in kwargs:
		relativeCameraPoseLoss_lossWeight = kwargs["relativeCameraPoseLoss_lossWeight"]
	else:
		relativeCameraPoseLoss_lossWeight = 1.0
  
	if use_relativeCameraPoseLoss:
		assert unfreeze_videoEncoder and (recog_arc in ["egovlp_v2"])

	if relativeCameraPoseLoss_coordsAsClasses:
		assert relativeCameraPoseLoss_rotationAsClasses
		assert relativeCameraPoseLoss_coordsInAngles
		assert relativeCameraPoseLoss_coordsClassSize > 0
		assert 360 % relativeCameraPoseLoss_coordsClassSize == 0
		assert 180 % relativeCameraPoseLoss_coordsClassSize == 0
		coorsAlpha_numClasses = int(360 // relativeCameraPoseLoss_coordsClassSize) + 1
		coorsBeta_numClasses = int(180 // relativeCameraPoseLoss_coordsClassSize) + 1
		coors_numClasses = coorsAlpha_numClasses + coorsBeta_numClasses

	if relativeCameraPoseLoss_rotationAsClasses:
		assert relativeCameraPoseLoss_rotationInAngles
		assert relativeCameraPoseLoss_rotationClassSize > 0
		assert 360 % relativeCameraPoseLoss_rotationClassSize == 0
		assert 180 % relativeCameraPoseLoss_rotationClassSize == 0
		rotsAngX_numClasses = rotsAngZ_numClasses = (int(360 // relativeCameraPoseLoss_rotationClassSize) + 1)
		rotsAngY_numClasses = int((180 // relativeCameraPoseLoss_rotationClassSize) + 1)
		rots_numClasses = rotsAngX_numClasses + rotsAngY_numClasses + rotsAngZ_numClasses

	use_egovlpV2_patchLevelVisualFeats = kwargs["use_egovlpV2_patchLevelVisualFeats"] if ("use_egovlpV2_patchLevelVisualFeats" in kwargs) else False
	egovlpV2_encodeWdinoV2 = kwargs["egovlpV2_encodeWdinoV2"] if ("egovlpV2_encodeWdinoV2" in kwargs) else False

	assert recog_arc in ["egovlp_v2",]
	if use_preExtractedFeats:
		vid_encoder = nn.Identity()
	else:
		vid_encoder = pol.videoEncoder(kwargs)
	model = pol.pol_v1(kwargs)

	if kwargs["distributed"]:
		vid_encoder = vid_encoder.to(device)
		model = model.to(device)

		if not use_preExtractedFeats:
			vid_encoder = DDP(vid_encoder, device_ids=[args.gpu],find_unused_parameters=True)
		model = DDP(model, device_ids=[args.gpu])
	else:
		vid_encoder = vid_encoder.to(device)
		model = model.to(device)
		if kwargs["data_parallel"]:
			assert n_available_gpus > 0
			print("Using", n_available_gpus, "GPUs!")
			vid_encoder = nn.DataParallel(vid_encoder, device_ids=list(range(n_available_gpus)), output_device=0)
			model = nn.DataParallel(model, device_ids=list(range(n_available_gpus)), output_device=0)

	if ("lr_videoEncoder" in kwargs) and unfreeze_videoEncoder and (kwargs["lr_videoEncoder"] != kwargs["lr"]):
		params2 = []
		if use_egovlpV2_patchLevelVisualFeats:
			params2 = list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.conv_egovlpV2_patchLevelVisualFeats.parameters()))
		else:
			if egovlpV2_encodeWdinoV2:
				params2 = list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.egovlpV2_encodeWdinoV2_x_norm_clstoken_agg.parameters()))

		if use_relativeCameraPoseLoss:
			params2 += list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.shared_conv.parameters())) +\
						list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.pose_conv.parameters())) +\
						list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.pose_lin.parameters())) +\
						list(filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.pred_conv.parameters()))

		params2 += list(filter(lambda p: p.requires_grad, model.parameters()))

		if optimizer_type == "adam_w":
			optimizer = optim.AdamW([{"params": filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.model.parameters()),
									  "lr": kwargs["lr_videoEncoder"]},
									 {"params": filter(lambda p: p.requires_grad, params2),
									 }
									],
									lr=kwargs["lr"],
									weight_decay=kwargs["weight_decay"] if kwargs["weight_decay"] else 0.)
		elif optimizer_type == "adam":
			optimizer = optim.Adam([{"params": filter(lambda p: p.requires_grad, vid_encoder.module.vid_encoder.model.parameters()),
									  "lr": kwargs["lr_videoEncoder"]},
									 {"params": filter(lambda p: p.requires_grad, params2),
									 }
									],
									lr=kwargs["lr"],
									weight_decay=kwargs["weight_decay"] if kwargs["weight_decay"] else 0.)
	else:
		if optimizer_type == "adam_w":
			optimizer = optim.AdamW(filter(lambda p: p.requires_grad, 
													(list(vid_encoder.parameters()) + list(model.parameters()))\
														if unfreeze_videoEncoder else model.parameters()
										),
									lr=kwargs["lr"],
									weight_decay=kwargs["weight_decay"] if kwargs["weight_decay"] else 0.)
		elif optimizer_type == "adam":
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
													(list(vid_encoder.parameters()) + list(model.parameters()))\
														if unfreeze_videoEncoder else model.parameters()
										),
									lr=kwargs["lr"],
									weight_decay=kwargs["weight_decay"] if kwargs["weight_decay"] else 0.)

	if not unfreeze_videoEncoder:
		vid_encoder.eval()

	start_epoch = 0
	min_loss = float('inf')
	max_acc = float('-inf')
	max_captioningScores = [float('-inf')] * (len(kwargs["valDatapoints_filePath"]) if isinstance(kwargs["valDatapoints_filePath"], list) else 1)
	if load_ckpt:
		min_loss, start_epoch =\
			loadModel_trainer(loaded_ckpt,
							  model,
							  vid_encoder=vid_encoder if unfreeze_videoEncoder else None,
							  optimizer=optimizer,
							  kwargs=kwargs)
		if isinstance(min_loss, tuple):
			if len(min_loss) == 2:
				max_acc = min_loss[0]
				min_loss = min_loss[1]
			elif len(min_loss) == 3:
				max_acc = min_loss[0]
				max_captioningScores = min_loss[1]
				min_loss = min_loss[2]
			else:
				raise NotImplementedError
		else:
			max_acc = min_loss

	if kwargs["distributed"]:
		dist.barrier()

		train_sampler = DistributedSampler(
		    train_loader,
		    shuffle=True,
		    num_replicas=get_world_size(args),
		    rank=get_rank(args),
		)
		val_sampler = DistributedSampler(
		    val_loader,
		    shuffle=False,
		    num_replicas=get_world_size(args),
		    rank=get_rank(args),
		)

		train_loader = DataLoader(
		    train_loader,
		    batch_size=kwargs["batch_size"],
		    num_workers=kwargs["num_workers"],
		    pin_memory=True,
		    sampler=train_sampler,
		    shuffle=False,
		    collate_fn=None,
		    drop_last=True,
		)
		val_loader = DataLoader(
		    val_loader,
		    batch_size=kwargs["batch_size"],
		    num_workers=kwargs["num_workers"],
		    pin_memory=True,
		    sampler=val_sampler,
		    shuffle=False,
		    collate_fn=None,
		    drop_last=False,
		)

		train_loader = PrefetchLoader(train_loader)
		val_loader = PrefetchLoader(val_loader)

		train_loader = IterLoader(train_loader, use_distributed=True)
		val_loader = IterLoader(val_loader, use_distributed=True)


	prev_val_loss = None
	# LR scheduler setup (per-iteration)
	use_lr_scheduler = kwargs.get("use_lr_scheduler", False)
	lr_warmup_iters = int(kwargs.get("lr_warmup_iters", 5000))
	lr_warmup_start = float(kwargs.get("lr_warmup_start", 1e-6))
	lr_min = float(kwargs.get("lr_min", 1e-5))
	lr_sched_total_iters_arg = int(kwargs.get("lr_scheduler_total_iters", 0))

	# Capture base LR per param group
	base_lrs = [pg.get("lr", kwargs["lr"]) for pg in optimizer.param_groups]

	def compute_lr(step, base_lr, total_iters):
		if step < lr_warmup_iters:
			# Linear warmup from lr_warmup_start to base_lr
			pct = 0.0 if lr_warmup_iters == 0 else float(step) / float(max(lr_warmup_iters, 1))
			return lr_warmup_start + (base_lr - lr_warmup_start) * pct
		# Cosine annealing to lr_min after warmup
		remain = max(total_iters - lr_warmup_iters, 1)
		prog = min(max((step - lr_warmup_iters) / remain, 0.0), 1.0)
		import math
		cos_term = 0.5 * (1.0 + math.cos(math.pi * prog))
		return lr_min + (base_lr - lr_min) * cos_term

	# Estimate iters per epoch and total iters if not provided
	if kwargs["distributed"]:
		iters_per_epoch_est = kwargs["num_trainIterations"]
	else:
		# When not distributed, train_loader is a DataLoader
		iters_per_epoch_est = len(train_loader)
		total_train_samples = kwargs.get("num_trainSamples", 0)
		if total_train_samples and kwargs.get("batch_size", 0):
			iters_per_epoch_est = max(total_train_samples // kwargs["batch_size"], 1)

	total_iters_est = lr_sched_total_iters_arg if lr_sched_total_iters_arg > 0 else max(num_epochs * iters_per_epoch_est, 1)

	# Initialize global step considering resume
	global_step = start_epoch * iters_per_epoch_est

	prev_val_loss = None
	for epoch in range(start_epoch, num_epochs):
		print(f"Epoch {epoch + 1} out of {num_epochs} epochs")
		if unfreeze_videoEncoder:
			vid_encoder.train()
		model.train()
		train_loss = 0.
		train_loss_relCameraPose = 0.
		train_acc = 0.
		train_acc_multiHot = 0. 
		train_captioningScores = [0.] * (len(kwargs["trainDatapoints_filePath"]) if isinstance(kwargs["trainDatapoints_filePath"], list) else 1)
		train_numSamples = train_numSamples_relCameraPose = 0
		if kwargs["distributed"]:
			if not hasattr(train_loader, "__next__"):
				train_loader = iter(train_loader)
			metric_logger = MetricLogger(delimiter="  ")
			metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			if use_relativeCameraPoseLoss:
				metric_logger.add_meter("loss_relCameraPose", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			metric_logger.add_meter("accuracy", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			if task_type in ["classify_oneHot", "match_dist"]:
				metric_logger.add_meter("accuracy_multiHot", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			for captioner_idx in range(len(train_captioningScores)):
				metric_logger.add_meter(f"captioning_score_{captioner_idx + 1}", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))

		for ele_idx, loader_ele in enumerate(tqdm(train_loader)):
			if kwargs["distributed"]:
				if ele_idx >= num_trainIters:
					break

			if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
				if use_relativeCameraPoseLoss:
					if egoVlpV2_vis2textSim_labler:
						frames, label, label_multiHot, captioning_scores, captioning_scores_actual, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
					else:
						frames, label, label_multiHot, captioning_scores, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
				else:
					if egoVlpV2_vis2textSim_labler:
						frames, label, label_multiHot, captioning_scores, captioning_scores_actual, class_wts = loader_ele
					else:
						frames, label, label_multiHot, captioning_scores, class_wts = loader_ele
			else:
				if use_relativeCameraPoseLoss:
					if egoVlpV2_vis2textSim_labler:
						frames, label, captioning_scores, captioning_scores_actual, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
					else:
						frames, label, captioning_scores, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
				else:
					if egoVlpV2_vis2textSim_labler:
						frames, label, captioning_scores, captioning_scores_actual, class_wts = loader_ele
					else:
						frames, label, captioning_scores, class_wts = loader_ele

			if kwargs["distributed"]:
				frames = prepare_sample(frames, cuda_enabled=device.type=="cuda")
				label = prepare_sample(label, cuda_enabled=device.type=="cuda")
				class_wts = prepare_sample(class_wts, cuda_enabled=device.type=="cuda")
				if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
					label_multiHot = prepare_sample(label_multiHot, cuda_enabled=device.type=="cuda")
				captioning_scores = prepare_sample(captioning_scores, cuda_enabled=device.type=="cuda")
				if egoVlpV2_vis2textSim_labler:
					captioning_scores_actual = prepare_sample(captioning_scores_actual, cuda_enabled=device.type=='cuda')
				if use_relativeCameraPoseLoss:
					gt_relCameraPose = prepare_sample(gt_relCameraPose, cuda_enabled=device.type=="cuda")
					has_relCameraPose = prepare_sample(has_relCameraPose, cuda_enabled=device.type=="cuda")
			else:
				frames = frames.to(device)
				label = label.to(device)
				class_wts = class_wts.to(device)
				if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
					label_multiHot = label_multiHot.to(device)
				captioning_scores = captioning_scores.to(device)
				if egoVlpV2_vis2textSim_labler:
					captioning_scores_actual = captioning_scores_actual.to(device)
				if use_relativeCameraPoseLoss:
					gt_relCameraPose = gt_relCameraPose.to(device)
					has_relCameraPose = has_relCameraPose.to(device)

			if use_relativeCameraPoseLoss:
				gt_relCameraPose_coords = None
				gt_relCameraPose_rots = None
				if relativeCameraPoseLoss_rotationOnly:	
					gt_relCameraPose_rots = gt_relCameraPose
				else:
					if relativeCameraPoseLoss_coordsInAngles:
						gt_relCameraPose_coords = gt_relCameraPose[..., :2]
						gt_relCameraPose_rots = gt_relCameraPose[..., 2:]
					else:
						gt_relCameraPose_coords = gt_relCameraPose[..., :3]
						gt_relCameraPose_rots = gt_relCameraPose[..., 3:]

				if relativeCameraPoseLoss_rotationAsClasses:
					gt_relCameraPose_rots = (gt_relCameraPose_rots // relativeCameraPoseLoss_rotationClassSize).long()

				if relativeCameraPoseLoss_coordsAsClasses:
					gt_relCameraPose_coords = (gt_relCameraPose_coords // relativeCameraPoseLoss_coordsClassSize).long()

			if (frame_colorJitter != [0, 0, 0]) and (frame_colorJitter != [0, 0]):
				frames = frames.permute((0,1,2,5,3,4))	# -> 4, 5, 8, 3, 224, 224
				frames = train_transforms_colorJitter.to(frames.device)(frames)
				frames = frames.permute((3, 0, 1, 2, 4, 5)) # -> 3, 4, 5, 8, 224, 224 
				bs, num_frames = frames.shape[1], frames.shape[2]
				frames = frames.reshape((frames.shape[0], 
										bs * num_frames * frames.shape[3], 
										frames.shape[4], 
										frames.shape[5]))
				frames = train_transforms_normalize(frames)
				frames = frames.reshape((frames.shape[0], bs, num_frames, -1, frames.shape[2], frames.shape[3]))	# -> 3, 4, 5, 8, 224, 224 
				frames = frames.permute((1, 2, 3, 4, 5, 0))

			if unfreeze_videoEncoder:
				if use_relativeCameraPoseLoss:
					feats, feats_relCameraPose = vid_encoder(frames)
				else:
					feats = vid_encoder(frames)
			else:
				with torch.no_grad():
					feats = vid_encoder(frames).detach()
			out = model(feats)
			if task_type in ["classify_oneHot", "classify_oneHot_bestExoPred"]:
				if use_minMultiHotLoss or use_randMultiHotLoss:
					for idx_label_multiHot, ele_label_multiHot in enumerate(label_multiHot):
						ele_labels_oneHot = torch.where(ele_label_multiHot == 1.0)[0].long()
						ele_losses = []
						for ele_label_oneHot in ele_labels_oneHot:
							ele_losses.append(F.cross_entropy(out[idx_label_multiHot: idx_label_multiHot + 1],
																ele_label_oneHot.unsqueeze(0) ))
						if use_minMultiHotLoss:
							min_idx = torch.argmin(torch.tensor(ele_losses)).item()
							if idx_label_multiHot == 0:
								if balanceCLasses_inLoss:
									loss = ele_losses[min_idx] *\
											(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[min_idx].int().item()].item())
								else:
									loss = ele_losses[min_idx]
							else:
								if balanceCLasses_inLoss:
									loss += (ele_losses[min_idx] *\
												(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[min_idx].int().item()].item()))
								else:
									loss += ele_losses[min_idx]
						elif use_randMultiHotLoss:
							rnd_idx = torch.randint(len(ele_losses), size=(1,)).item()
							if idx_label_multiHot == 0:
								if balanceCLasses_inLoss:
									loss = ele_losses[rnd_idx] *\
											(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[rnd_idx].int().item()].item())
								else:
									loss = ele_losses[rnd_idx]
							else:
								if balanceCLasses_inLoss:
									loss += (ele_losses[rnd_idx] *\
												(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[rnd_idx].int().item()].item()))
								else:
									loss += ele_losses[rnd_idx]

					loss /= len(label_multiHot)
				elif use_bceMultiHotLoss:
					loss = F.binary_cross_entropy(torch.sigmoid(out), label_multiHot,)
				elif use_klLoss:
					captioning_scores_label = torch.mean(F.softmax(captioning_scores, dim=2), dim=1)
					loss = F.kl_div(F.log_softmax(out, dim=1), captioning_scores_label, reduction='batchmean')
				else:
					if len(label.shape) == 2:
						label = label.squeeze(-1)

					loss = F.cross_entropy(out, label,) 
			elif task_type == 'match_dist':
				if balanceCLasses_inLoss:
					raise NotImplementedError
				loss = F.kl_div(F.log_softmax(out, dim=1), F.softmax(label, dim=1), reduction='batchmean')
			elif task_type == "classify_multiHot_bestExoPred":
				if balanceCLasses_inLoss:
					raise NotImplementedError
				loss = F.binary_cross_entropy(torch.sigmoid(out), label,)

			if use_relativeCameraPoseLoss:
				if relativeCameraPoseLoss_rotationAsClasses:
					if relativeCameraPoseLoss_coordsAsClasses:
						feats_relCameraPose_coords = feats_relCameraPose[..., :coors_numClasses]

						gt_relCameraPose_coordsAlpha = gt_relCameraPose_coords[..., 0]
						feats_relCameraPose_coordsAlpha = feats_relCameraPose_coords[..., :coorsAlpha_numClasses]
						loss_relCameraPose_coordsAlpha = F.cross_entropy(feats_relCameraPose_coordsAlpha.reshape(-1, feats_relCameraPose_coordsAlpha.shape[-1]), 
																			gt_relCameraPose_coordsAlpha.reshape(-1),
																			reduction="none",) 

						gt_relCameraPose_coordsBeta = gt_relCameraPose_coords[..., 1]
						feats_relCameraPose_coordsBeta = feats_relCameraPose_coords[..., coorsAlpha_numClasses:]
						loss_relCameraPose_coordsBeta = F.cross_entropy(feats_relCameraPose_coordsBeta.reshape(-1, feats_relCameraPose_coordsBeta.shape[-1]), 
																		gt_relCameraPose_coordsBeta.reshape(-1),
																		reduction="none",) 

						feats_relCameraPose_rots = feats_relCameraPose[..., coors_numClasses:]
					else:
						raise NotImplementedError

					gt_relCameraPose_rotsAngX = gt_relCameraPose_rots[..., 0]
					feats_relCameraPose_rotsAngX = feats_relCameraPose_rots[..., :rotsAngX_numClasses]
					loss_relCameraPose_rotsAngX = F.cross_entropy(feats_relCameraPose_rotsAngX.reshape(-1, feats_relCameraPose_rotsAngX.shape[-1]), 
																	gt_relCameraPose_rotsAngX.reshape(-1),
																	reduction="none",) 

					gt_relCameraPose_rotsAngY = gt_relCameraPose_rots[..., 1]
					feats_relCameraPose_rotsAngY = feats_relCameraPose_rots[..., rotsAngX_numClasses: rotsAngX_numClasses + rotsAngY_numClasses]
					loss_relCameraPose_rotsAngY = F.cross_entropy(feats_relCameraPose_rotsAngY.reshape(-1, feats_relCameraPose_rotsAngY.shape[-1]), 
																	gt_relCameraPose_rotsAngY.reshape(-1),
																	reduction="none",) 

					gt_relCameraPose_rotsAngZ = gt_relCameraPose_rots[..., 2]
					feats_relCameraPose_rotsAngZ = feats_relCameraPose_rots[..., rotsAngX_numClasses + rotsAngY_numClasses:]
					loss_relCameraPose_rotsAngZ = F.cross_entropy(feats_relCameraPose_rotsAngZ.reshape(-1, feats_relCameraPose_rotsAngZ.shape[-1]), 
																	gt_relCameraPose_rotsAngZ.reshape(-1),
																	reduction="none",) 

					if relativeCameraPoseLoss_coordsAsClasses:
						loss_relCameraPose = loss_relCameraPose_coordsAlpha + loss_relCameraPose_coordsBeta +\
												loss_relCameraPose_rotsAngX + loss_relCameraPose_rotsAngY + loss_relCameraPose_rotsAngZ
						loss_relCameraPose = loss_relCameraPose / 5
						loss_relCameraPose = loss_relCameraPose.reshape((gt_relCameraPose_coordsAlpha.shape[0],
																		gt_relCameraPose_coordsAlpha.shape[1],
																		gt_relCameraPose_coordsAlpha.shape[2]))	
					else:
						loss_relCameraPose = loss_relCameraPose_coords +loss_relCameraPose_rotsAngX +\
											loss_relCameraPose_rotsAngY + loss_relCameraPose_rotsAngZ
						loss_relCameraPose = loss_relCameraPose / 4
						loss_relCameraPose = loss_relCameraPose.reshape((gt_relCameraPose_coords.shape[0],
																		gt_relCameraPose_coords.shape[1],
																		gt_relCameraPose_coords.shape[2]))	
				elif relativeCameraPoseLoss_lossType == "l2":
					loss_relCameraPose =  F.mse_loss(feats_relCameraPose, gt_relCameraPose, reduction="none")
				elif relativeCameraPoseLoss_lossType == "l1":
					loss_relCameraPose =  F.l1_loss(feats_relCameraPose, gt_relCameraPose, reduction="none")
				else:
					raise ValueError

				loss_relCameraPose = loss_relCameraPose.reshape((loss_relCameraPose.shape[0], -1)) * has_relCameraPose.unsqueeze(1)
				loss_relCameraPose = torch.sum(loss_relCameraPose) / (max(torch.sum(has_relCameraPose).item(), 1) * loss_relCameraPose.shape[1])

				total_loss = loss + relativeCameraPoseLoss_lossWeight * loss_relCameraPose
			else:
				total_loss = loss

			optimizer.zero_grad()
			total_loss.backward()	
			optimizer.step()

			# Scheduler step per iteration
			if use_lr_scheduler:
				for pg_idx, pg in enumerate(optimizer.param_groups):
					pg["lr"] = compute_lr(global_step, base_lrs[pg_idx], total_iters_est)
				global_step += 1

			if task_type in ["classify_oneHot", "match_dist",]: 
				if task_type in ["match_dist",]:
					label = torch.argmax(label, dim=1).long()
				train_acc_thisBatch = int(torch.sum(torch.argmax(out, dim=1).long() == label))

			if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
				out_oneHot = torch.zeros((out.shape[0], out.shape[1])).to(out.device)
				out_oneHot[list(range(len(out_oneHot))), torch.argmax(out, dim=1).detach().tolist()] = 1
				train_acc_multiHot_thisBatch = len(set(torch.where((out_oneHot + label_multiHot) == 2.0)[0].detach().tolist()))

				assert train_acc_multiHot_thisBatch <= len(out_oneHot)

				if task_type in ["classify_oneHot_bestExoPred"]:
					train_acc_thisBatch = train_acc_multiHot_thisBatch
			elif task_type == "classify_multiHot_bestExoPred":
				out_multiHot = (out > 0.5).float()
				train_acc_thisBatch= len(set(torch.where((out_multiHot + label) == 2.0)[0].detach().tolist()))
				assert train_acc_thisBatch <= len(out)

			train_captioningScores_thisBatch = []
			for captioner_idx in range(len(train_captioningScores)):
				if egoVlpV2_vis2textSim_labler:
					train_captioningScores_thisBatch.append(
							torch.sum(captioning_scores_actual[:, captioner_idx][list(range(len(captioning_scores_actual[:, captioner_idx]))), 
																				torch.argmax(out, dim=1).detach().tolist()]).item()
						)
				else:
					train_captioningScores_thisBatch.append(
							torch.sum(captioning_scores[:, captioner_idx][list(range(len(captioning_scores[:, captioner_idx]))), 
																				torch.argmax(out, dim=1).detach().tolist()]).item()
						)

			if kwargs["distributed"]:
				metric_logger.update(len(label), loss=loss.item())
				if use_relativeCameraPoseLoss:
					metric_logger.update(torch.sum(has_relCameraPose).item(), loss_relCameraPose=loss_relCameraPose.item())
				metric_logger.update(len(label), accuracy=train_acc_thisBatch / len(label))
				if task_type in ["classify_oneHot", "match_dist"]:
					metric_logger.update(len(label), accuracy_multiHot=train_acc_multiHot_thisBatch / len(label))
				for captioner_idx in range(len(train_captioningScores)):
					if captioner_idx == 0:
						metric_logger.update(len(label),
											captioning_score_1=train_captioningScores_thisBatch[captioner_idx] / len(label))
					elif captioner_idx == 1:
						metric_logger.update(len(label),
											captioning_score_2=train_captioningScores_thisBatch[captioner_idx] / len(label))
					elif captioner_idx == 2:
						metric_logger.update(len(label),
											captioning_score_3=train_captioningScores_thisBatch[captioner_idx] / len(label))
					else:
						raise ValueError
			else:
				train_acc += train_acc_thisBatch
				if task_type in ["classify_oneHot", "match_dist"]:
					train_acc_multiHot += train_acc_multiHot_thisBatch
				train_loss += loss.item() * len(label)
				if use_relativeCameraPoseLoss:
					train_loss_relCameraPose += loss_relCameraPose.item() * torch.sum(has_relCameraPose).item()
				for captioner_idx in range(len(train_captioningScores)):
					train_captioningScores[captioner_idx] += train_captioningScores_thisBatch[captioner_idx]
				train_numSamples += len(label)
				if use_relativeCameraPoseLoss:
					train_numSamples_relCameraPose += torch.sum(has_relCameraPose).item()

		if kwargs["distributed"]:
			metric_logger.synchronize_between_processes()
			train_loss = metric_logger.meters["loss"].global_avg
			if use_relativeCameraPoseLoss:
				train_loss_relCameraPose = metric_logger.meters["loss_relCameraPose"].global_avg
			train_acc = metric_logger.meters["accuracy"].global_avg
			if task_type in ["classify_oneHot", "match_dist"]:
				train_acc_multiHot = metric_logger.meters["accuracy_multiHot"].global_avg
			train_captioningScores_tmp = []
			for captioner_idx in range(len(train_captioningScores)):
				train_captioningScores_tmp.append(metric_logger.meters[f"captioning_score_{captioner_idx + 1}"].global_avg)
			train_captioningScores = train_captioningScores_tmp
		else:
			train_loss /= max(train_numSamples, 1)
			if use_relativeCameraPoseLoss:
				train_loss_relCameraPose /= max(train_numSamples_relCameraPose, 1)
			train_acc /= max(train_numSamples, 1)
			if task_type in ["classify_oneHot", "match_dist"]:
				train_acc_multiHot /= max(train_numSamples, 1)
			for captioner_idx in range(len(train_captioningScores)):
				train_captioningScores[captioner_idx] = train_captioningScores[captioner_idx] / max(train_numSamples, 1)

		if task_type in ["classify_oneHot", "match_dist"]:
			if use_relativeCameraPoseLoss:
				print(f"Train: loss -- {train_loss:.4f}, loss_relCameraPose -- {train_loss_relCameraPose:.4f}, "+\
						f"accuracy -- {train_acc:.4f}, accuracy multi-hot -- {train_acc_multiHot:.4f}, "+\
						f"captioning scores -- {[round(train_captioningScore, 4) for train_captioningScore in train_captioningScores]}, ")
			else:
				print(f"Train: loss -- {train_loss:.4f}, accuracy -- {train_acc:.4f}, "+\
						f"accuracy multi-hot -- {train_acc_multiHot:.4f}, captioning score -- {[round(train_captioningScore, 4) for train_captioningScore in train_captioningScores]}")
		else:
			if use_relativeCameraPoseLoss:
				print(f"Train: loss -- {train_loss:.4f}, loss_relCameraPose -- {train_loss_relCameraPose:.4f}, "+\
					  f"accuracy -- {train_acc:.4f}, captioning score -- {[round(train_captioningScore, 4) for train_captioningScore in train_captioningScores]}")
			else:
				print(f"Train: loss -- {train_loss:.4f}, accuracy -- {train_acc:.4f}, captioning score -- {[round(train_captioningScore, 4) for train_captioningScore in train_captioningScores]}")

		if unfreeze_videoEncoder:
			vid_encoder.eval()
		model.eval()
		val_loss = 0.
		val_loss_relCameraPose = 0.
		val_acc = 0.
		val_acc_multiHot = 0.
		val_captioningScores = [0.] * (len(kwargs["valDatapoints_filePath"]) if isinstance(kwargs["valDatapoints_filePath"], list) else 1)
		val_numSamples = val_numSamples_relCameraPose = 0
		if kwargs["distributed"]:
			if not hasattr(val_loader, "__next__"):
				val_loader = iter(val_loader)
			metric_logger = MetricLogger(delimiter="  ")
			metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				metric_logger.add_meter("loss_relCameraPose", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			metric_logger.add_meter("accuracy", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			if task_type in ["classify_oneHot", "match_dist"]:
				metric_logger.add_meter("accuracy_multiHot", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
			for captioner_idx in range(len(val_captioningScores)):
				metric_logger.add_meter(f"captioning_score_{captioner_idx + 1}", SmoothedValue(window_size=1, fmt="{value:.4f}", args=args))
		for ele_idx, loader_ele in enumerate(tqdm(val_loader)):
			if kwargs["distributed"]:
				if ele_idx >= num_valIters:
					break
			if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					if egoVlpV2_vis2textSim_labler:
						frames, label, label_multiHot, captioning_scores, captioning_scores_actual, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
					else:
						frames, label, label_multiHot, captioning_scores, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
				else:
					if egoVlpV2_vis2textSim_labler:
						frames, label, label_multiHot, captioning_scores, captioning_scores_actual, class_wts = loader_ele
					else:
						frames, label, label_multiHot, captioning_scores, class_wts = loader_ele
			else:
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					if egoVlpV2_vis2textSim_labler:
						frames, label, captioning_scores, captioning_scores_actual, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
					else:
						frames, label, captioning_scores, class_wts, gt_relCameraPose, has_relCameraPose = loader_ele
				else:
					if egoVlpV2_vis2textSim_labler:
						frames, label, captioning_scores, captioning_scores_actual, class_wts = loader_ele
					else:
						frames, label, captioning_scores, class_wts = loader_ele

			if kwargs["distributed"]:
				frames = prepare_sample(frames, cuda_enabled=device.type=="cuda")
				label = prepare_sample(label, cuda_enabled=device.type=="cuda")
				class_wts = prepare_sample(class_wts, cuda_enabled=device.type=="cuda")
				if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
					label_multiHot = prepare_sample(label_multiHot, cuda_enabled=device.type=="cuda")
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					gt_relCameraPose = prepare_sample(gt_relCameraPose, cuda_enabled=device.type=="cuda")
					has_relCameraPose = prepare_sample(has_relCameraPose, cuda_enabled=device.type=="cuda")
				captioning_scores = prepare_sample(captioning_scores, cuda_enabled=device.type=="cuda")
				if egoVlpV2_vis2textSim_labler:
					captioning_scores_actual = prepare_sample(captioning_scores_actual, cuda_enabled=device.type=="cuda")
			else:
				frames = frames.to(device)
				label = label.to(device)
				class_wts = class_wts.to(device)
				if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
					label_multiHot = label_multiHot.to(device)
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					gt_relCameraPose = gt_relCameraPose.to(device)
					has_relCameraPose = has_relCameraPose.to(device)
				captioning_scores =  captioning_scores.to(device)
				if egoVlpV2_vis2textSim_labler:
					captioning_scores_actual = captioning_scores_actual.to(device)

			if use_relativeCameraPoseLoss:
				gt_relCameraPose_coords = None
				gt_relCameraPose_rots = None
				if relativeCameraPoseLoss_rotationOnly:	
					gt_relCameraPose_rots = gt_relCameraPose
				else:
					if relativeCameraPoseLoss_coordsInAngles:
						gt_relCameraPose_coords = gt_relCameraPose[..., :2]
						gt_relCameraPose_rots = gt_relCameraPose[..., 2:]
					else:
						gt_relCameraPose_coords = gt_relCameraPose[..., :3]
						gt_relCameraPose_rots = gt_relCameraPose[..., 3:]

				if relativeCameraPoseLoss_rotationAsClasses:
					gt_relCameraPose_rots = (gt_relCameraPose_rots // relativeCameraPoseLoss_rotationClassSize).long()

				if relativeCameraPoseLoss_coordsAsClasses:
					gt_relCameraPose_coords = (gt_relCameraPose_coords // relativeCameraPoseLoss_coordsClassSize).long()

			with torch.no_grad():
				if use_relativeCameraPoseLoss:
					feats, feats_relCameraPose = vid_encoder(frames)
					feats_relCameraPose = feats_relCameraPose.detach()
				else:
					feats = vid_encoder(frames)
				feats = feats.detach()
				out = model(feats)

			if task_type in ["classify_oneHot", "classify_oneHot_bestExoPred"]:
				if use_minMultiHotLoss or use_randMultiHotLoss:
					for idx_label_multiHot, ele_label_multiHot in enumerate(label_multiHot):
						ele_labels_oneHot = torch.where(ele_label_multiHot == 1.0)[0].long()
						ele_losses = []
						for ele_label_oneHot in ele_labels_oneHot:
							ele_losses.append(F.cross_entropy(out[idx_label_multiHot: idx_label_multiHot + 1],
																ele_label_oneHot.unsqueeze(0) ))
						if idx_label_multiHot == 0:
							if balanceCLasses_inLoss:
								loss = ele_losses[torch.argmin(torch.tensor(ele_losses)).item()] *\
										(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[torch.argmin(torch.tensor(ele_losses)).item()].int().item()].item())
							else:
								loss = ele_losses[torch.argmin(torch.tensor(ele_losses)).item()]
						else:
							if balanceCLasses_inLoss:
								loss += (ele_losses[torch.argmin(torch.tensor(ele_losses)).item()] *\
											(torch.min(class_wts[idx_label_multiHot]).item() / class_wts[idx_label_multiHot][ele_labels_oneHot[torch.argmin(torch.tensor(ele_losses)).item()].int().item()].item()))
							else:
								loss += ele_losses[torch.argmin(torch.tensor(ele_losses)).item()]

					loss /= len(label_multiHot)
				elif use_bceMultiHotLoss:
					loss = F.binary_cross_entropy(torch.sigmoid(out), label_multiHot,)
				elif use_klLoss:
					captioning_scores_label = torch.mean(F.softmax(captioning_scores, dim=2), dim=1)
					loss = F.kl_div(F.log_softmax(out, dim=1), captioning_scores_label, reduction='batchmean')
				else:
					if len(label.shape) == 2:
						label = label.squeeze(-1)
					loss = F.cross_entropy(out, label,) 
			elif task_type == 'match_dist':
				if balanceCLasses_inLoss:
					raise NotImplementedError
				loss = F.kl_div(F.log_softmax(out, dim=1), F.softmax(label, dim=1), reduction='batchmean')
			elif task_type == "classify_multiHot_bestExoPred":
				if balanceCLasses_inLoss:
					raise NotImplementedError
				loss = F.binary_cross_entropy(torch.sigmoid(out), label,)

			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				if relativeCameraPoseLoss_rotationAsClasses:
					feats_relCameraPose_coords = feats_relCameraPose[..., :coors_numClasses]

					gt_relCameraPose_coordsAlpha = gt_relCameraPose_coords[..., 0]
					feats_relCameraPose_coordsAlpha = feats_relCameraPose_coords[..., :coorsAlpha_numClasses]
					loss_relCameraPose_coordsAlpha = F.cross_entropy(feats_relCameraPose_coordsAlpha.reshape(-1, feats_relCameraPose_coordsAlpha.shape[-1]), 
																		gt_relCameraPose_coordsAlpha.reshape(-1),
																		reduction="none",) 

					gt_relCameraPose_coordsBeta = gt_relCameraPose_coords[..., 1]
					feats_relCameraPose_coordsBeta = feats_relCameraPose_coords[..., coorsAlpha_numClasses:]
					loss_relCameraPose_coordsBeta = F.cross_entropy(feats_relCameraPose_coordsBeta.reshape(-1, feats_relCameraPose_coordsBeta.shape[-1]), 
																	gt_relCameraPose_coordsBeta.reshape(-1),
																	reduction="none",) 


					feats_relCameraPose_rots = feats_relCameraPose[..., coors_numClasses:]

					gt_relCameraPose_rotsAngX = gt_relCameraPose_rots[..., 0]
					feats_relCameraPose_rotsAngX = feats_relCameraPose_rots[..., :rotsAngX_numClasses]
					loss_relCameraPose_rotsAngX = F.cross_entropy(feats_relCameraPose_rotsAngX.reshape(-1, feats_relCameraPose_rotsAngX.shape[-1]), 
																	gt_relCameraPose_rotsAngX.reshape(-1),
																	reduction="none",) 

					gt_relCameraPose_rotsAngY = gt_relCameraPose_rots[..., 1]
					feats_relCameraPose_rotsAngY = feats_relCameraPose_rots[..., rotsAngX_numClasses: rotsAngX_numClasses + rotsAngY_numClasses]
					loss_relCameraPose_rotsAngY = F.cross_entropy(feats_relCameraPose_rotsAngY.reshape(-1, feats_relCameraPose_rotsAngY.shape[-1]), 
																	gt_relCameraPose_rotsAngY.reshape(-1),
																	reduction="none",) 

					gt_relCameraPose_rotsAngZ = gt_relCameraPose_rots[..., 2]
					feats_relCameraPose_rotsAngZ = feats_relCameraPose_rots[..., rotsAngX_numClasses + rotsAngY_numClasses:]
					loss_relCameraPose_rotsAngZ = F.cross_entropy(feats_relCameraPose_rotsAngZ.reshape(-1, feats_relCameraPose_rotsAngZ.shape[-1]), 
																	gt_relCameraPose_rotsAngZ.reshape(-1),
																	reduction="none",) 


					loss_relCameraPose = loss_relCameraPose_coordsAlpha + loss_relCameraPose_coordsBeta +\
											loss_relCameraPose_rotsAngX + loss_relCameraPose_rotsAngY + loss_relCameraPose_rotsAngZ
					loss_relCameraPose = loss_relCameraPose / 5
					loss_relCameraPose = loss_relCameraPose.reshape((gt_relCameraPose_coordsAlpha.shape[0],
																	gt_relCameraPose_coordsAlpha.shape[1],
																	gt_relCameraPose_coordsAlpha.shape[2]))		
				elif relativeCameraPoseLoss_lossType == "l2":
					loss_relCameraPose =  F.mse_loss(feats_relCameraPose, gt_relCameraPose, reduction='none')
				elif relativeCameraPoseLoss_lossType == "l1":
					loss_relCameraPose =  F.l1_loss(feats_relCameraPose, gt_relCameraPose, reduction='none')
				else:
					raise ValueError

				loss_relCameraPose = loss_relCameraPose.reshape((loss_relCameraPose.shape[0], -1)) * has_relCameraPose.unsqueeze(1)
				loss_relCameraPose = torch.sum(loss_relCameraPose) / (max(torch.sum(has_relCameraPose).item(), 1) * loss_relCameraPose.shape[1])

			if task_type in ["classify_oneHot", "match_dist",]:
				if task_type in ["match_dist",]:
					label = torch.argmax(label, dim=1).long()
				val_acc_thisBatch = int(torch.sum(torch.argmax(out, dim=1).long() == label))

			if task_type in ["classify_oneHot", "match_dist", "classify_oneHot_bestExoPred"]:
				out_oneHot = torch.zeros((out.shape[0], out.shape[1])).to(out.device)
				out_oneHot[list(range(len(out_oneHot))), torch.argmax(out, dim=1).detach().tolist()] = 1
				val_acc_multiHot_thisBatch = len(set(torch.where((out_oneHot + label_multiHot) == 2.0)[0].detach().tolist()))

				if task_type in ["classify_oneHot_bestExoPred"]:
					val_acc_thisBatch = val_acc_multiHot_thisBatch
			elif task_type == "classify_multiHot_bestExoPred":
				out_multiHot = (out > 0.5).float()
				val_acc_thisBatch = len(set(torch.where((out_multiHot + label) == 2.0)[0].detach().tolist()))

			val_captioningScores_thisBatch = []
			for captioner_idx in range(len(val_captioningScores)):
				if egoVlpV2_vis2textSim_labler:
					val_captioningScores_thisBatch.append(
							torch.sum(captioning_scores_actual[:, captioner_idx][list(range(len(captioning_scores_actual[:, captioner_idx]))), 
																				torch.argmax(out, dim=1).detach().tolist()]).item()
						)
				else:
					val_captioningScores_thisBatch.append(
							torch.sum(captioning_scores[:, captioner_idx][list(range(len(captioning_scores[:, captioner_idx]))), 
																				torch.argmax(out, dim=1).detach().tolist()]).item()
						)

			if kwargs["distributed"]:
				metric_logger.update(len(label), loss=loss.item())
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					metric_logger.update(torch.sum(has_relCameraPose).item(), loss_relCameraPose=loss_relCameraPose.item())
				metric_logger.update(len(label), accuracy=val_acc_thisBatch / len(label))
				if task_type in ["classify_oneHot", "match_dist",]:
					metric_logger.update(len(label), accuracy_multiHot=val_acc_multiHot_thisBatch / len(label))
				for captioner_idx in range(len(val_captioningScores)):
					if captioner_idx == 0:
						metric_logger.update(len(label), captioning_score_1=val_captioningScores_thisBatch[captioner_idx] / len(label))
					elif captioner_idx == 1:
						metric_logger.update(len(label), captioning_score_2=val_captioningScores_thisBatch[captioner_idx] / len(label))
					elif captioner_idx == 2:
						metric_logger.update(len(label), captioning_score_3=val_captioningScores_thisBatch[captioner_idx] / len(label))
					else:
						raise NotImplementedError
			else:
				val_acc += val_acc_thisBatch 
				if task_type in ["classify_oneHot", "match_dist",]:
					val_acc_multiHot += val_acc_multiHot_thisBatch
				val_loss += loss.item() * len(label)
				if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
					val_loss_relCameraPose += loss_relCameraPose.item() * torch.sum(has_relCameraPose).item()
				for captioner_idx in range(len(val_captioningScores)):
					val_captioningScores[captioner_idx] += val_captioningScores_thisBatch[captioner_idx]
				val_numSamples += len(label)
				if use_relativeCameraPoseLoss:
					val_numSamples_relCameraPose += torch.sum(has_relCameraPose).item()

		if kwargs["distributed"]:
			dist.barrier()
			metric_logger.synchronize_between_processes()
			val_loss = metric_logger.meters["loss"].global_avg
			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				val_loss_relCameraPose = metric_logger.meters["loss_relCameraPose"].global_avg
			val_acc = metric_logger.meters["accuracy"].global_avg
			if task_type in ["classify_oneHot", "match_dist",]:
				val_acc_multiHot = metric_logger.meters["accuracy_multiHot"].global_avg
			val_captioningScores_tmp = []
			for captioner_idx in range(len(val_captioningScores)):
				val_captioningScores_tmp.append( metric_logger.meters[f"captioning_score_{captioner_idx + 1}"].global_avg)	
			val_captioningScores = val_captioningScores_tmp
		else:

			val_loss /= max(val_numSamples, 1)
			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				val_loss_relCameraPose /= max(val_numSamples_relCameraPose, 1)
			val_acc /= max(val_numSamples, 1)
			if task_type in ["classify_oneHot", "match_dist",]:
				val_acc_multiHot /= max(val_numSamples, 1)
			for captioner_idx in range(len(val_captioningScores)):
				val_captioningScores[captioner_idx] = val_captioningScores[captioner_idx] / max(val_numSamples, 1)	

		if task_type in ["classify_oneHot", "match_dist",]:
			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				print(f"Val: loss -- {val_loss:.4f}, loss_relCameraPose -- {val_loss_relCameraPose:.4f}, "+\
						f"accuracy -- {val_acc:.4f}, accuracy multi-hot -- {val_acc_multiHot:.4f}, "+
						f"captioning score -- {[round(val_captioningScore, 4) for val_captioningScore in val_captioningScores]}")
			else:
				print(f"Val: loss -- {val_loss:.4f}, accuracy -- {val_acc:.4f} "+\
						f"accuracy multi-hot -- {val_acc_multiHot:.4f}, captioning score -- {[round(val_captioningScore, 4) for val_captioningScore in val_captioningScores]}")
		else:
			if use_relativeCameraPoseLoss:	# use_relativeCameraPoseLoss / False
				print(f"Val: loss -- {val_loss:.4f}, loss_relCameraPose -- {val_loss_relCameraPose:.4f},"+\
						f" accuracy -- {val_acc:.4f}m captioning score -- {[round(val_captioningScore, 4) for val_captioningScore in val_captioningScores]}") 
			else:
				print(f"Val: loss -- {val_loss:.4f}, accuracy -- {val_acc:.4f}, captioning score -- {[round(val_captioningScore, 4) for val_captioningScore in val_captioningScores]}")

		is_best = False
		is_bestCaptioningScores = [False] * (len(kwargs["valDatapoints_filePath"]) if isinstance(kwargs["valDatapoints_filePath"], list) else 1)
		is_bestLoss = False
		if is_main_process(args):
			if task_type in ["classify_oneHot", "match_dist",]: 
				if val_acc_multiHot > max_acc:
					is_best = True
					max_acc = val_acc_multiHot
			else:
				if val_acc > max_acc:
					is_best = True
					max_acc = val_acc

			for captioner_idx, (val_captioningScore, max_captioningScore) in enumerate(zip(val_captioningScores, max_captioningScores)):
				if val_captioningScore > max_captioningScore:
					is_bestCaptioningScores[captioner_idx] = True
					max_captioningScores[captioner_idx] = val_captioningScore

			if val_loss < min_loss:
				is_bestLoss = True
				min_loss = val_loss

			if writer is not None:
				writer.add_scalar(f'Loss/train', train_loss, epoch)
				writer.add_scalar(f'Loss/val', val_loss, epoch)
				if use_relativeCameraPoseLoss:
					writer.add_scalar('loss_relCameraPose/train', train_loss_relCameraPose, epoch)
					writer.add_scalar('loss_relCameraPose/val', val_loss_relCameraPose, epoch)
				writer.add_scalar(f'Accuracy_firstGtOneHot/train', train_acc, epoch)
				writer.add_scalar(f'Accuracy_firstGtOneHot/val', val_acc, epoch)
				if task_type in ["classify_oneHot", "match_dist",]:
					writer.add_scalar(f'Accuracy_multiHot/train', train_acc_multiHot, epoch)
					writer.add_scalar(f'Accuracy_multiHot/val', val_acc_multiHot, epoch)
				for captioner_idx, (train_captioningScore, val_captioningScore) in enumerate(zip(train_captioningScores, val_captioningScores)):
					writer.add_scalar(f'Captionining_score_{captioner_idx}/train', train_captioningScore, epoch)
					writer.add_scalar(f'Captionining_score_{captioner_idx}/val', val_captioningScore, epoch)

			saveModel_trainer(kwargs,
							  ckpt_dir,
							  epoch,
							  model,
							  optimizer,
							  video_encoder=vid_encoder if unfreeze_videoEncoder else None,
							  best_metric=max_acc,
							  is_best=is_best,
							  task_type=task_type,
							  best_loss=min_loss,
							  is_bestLoss=is_bestLoss,
							  best_captioningScores=max_captioningScores,
							  is_bestCaptioningScores=is_bestCaptioningScores,
							  )
		print("-" * 80)

		# Early stopping: stop once validation loss starts increasing
		if prev_val_loss is not None and (val_loss > prev_val_loss):
			print("Validation loss increased. Early stopping.")
			break
		prev_val_loss = val_loss


def test(test_loader,
		 **kwargs):
	run_dir = kwargs["run_dir"]

	ckpt_dir = os.path.join(run_dir, "data")
	assert os.path.isdir(ckpt_dir)

	checkpoint_fileName = kwargs["checkpoint_fileName"] if ("checkpoint_fileName" in kwargs) else "valBestCkpt_maxCaptioningScore_captioner1"
	assert os.path.isfile(os.path.join(run_dir, f"data/{checkpoint_fileName}.pth"))
	loaded_ckpt = torch.load(os.path.join(run_dir,
										  f"data/{checkpoint_fileName}.pth"),
							 map_location="cpu")

	isLemma_dataset = kwargs["isLemma_dataset"] if ("isLemma_dataset" in kwargs) else False

	recog_arc = kwargs["recog_arc"]

	dump_fp = f"{run_dir}/test_index2logits_checkpoint-{checkpoint_fileName.split('_')[-1]}.json"

	batch_size = kwargs["batch_size"]

	device = (
		torch.device("cuda", 0)
		if torch.cuda.is_available()
		else torch.device("cpu")
	)
	n_available_gpus = torch.cuda.device_count()

	task_type = kwargs['task_type']
	unfreeze_videoEncoder = kwargs["unfreeze_videoEncoder"] if ("unfreeze_videoEncoder" in kwargs) else False

	assert task_type in ['classify_oneHot', 'match_dist', 'classify_oneHot_bestExoPred', 'classify_multiHot_bestExoPred']
	task_isBestExoPred = task_type in ['classify_oneHot_bestExoPred', 'classify_multiHot_bestExoPred']

	use_egoVlpV2_takeVideoFeats_usingStartNendTime = kwargs["use_egoVlpV2_takeVideoFeats_usingStartNendTime"]\
														if ("use_egoVlpV2_takeVideoFeats_usingStartNendTime" in kwargs) else\
															False
	use_egoVlpV2_takeVideoFeats_usingCenterTime = kwargs["use_egoVlpV2_takeVideoFeats_usingCenterTime"]\
														if ("use_egoVlpV2_takeVideoFeats_usingCenterTime" in kwargs) else\
															False
	use_egoVlpV2_takeVideoFeats = use_egoVlpV2_takeVideoFeats_usingStartNendTime or\
									use_egoVlpV2_takeVideoFeats_usingCenterTime	

	use_videoLlama_feats = kwargs["use_videoLlama_feats"] if ("use_videoLlama_feats" in kwargs) else False

	use_preExtractedFeats = use_egoVlpV2_takeVideoFeats or\
								use_videoLlama_feats 

	use_relativeCameraPoseLoss = kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False	

	assert recog_arc in ["egovlp_v2",]
	if use_preExtractedFeats:
		vid_encoder = nn.Identity()
	else:
		vid_encoder = pol.videoEncoder(kwargs)
	model = pol.pol_v1(kwargs)

	vid_encoder = vid_encoder.to(device)
	model = model.to(device)
	if kwargs["data_parallel"]:
		assert n_available_gpus > 0
		print("Using", n_available_gpus, "GPUs!")
		vid_encoder = nn.DataParallel(vid_encoder, device_ids=list(range(n_available_gpus)), output_device=0)
		model = nn.DataParallel(model, device_ids=list(range(n_available_gpus)), output_device=0)

	vid_encoder.eval()
	model.eval()
	
	loadModel_trainer(loaded_ckpt,
					  model,
					  vid_encoder=vid_encoder if unfreeze_videoEncoder else None,
					  kwargs=kwargs,
					  is_test=True)

	test_loss = 0.
	test_acc = 0.
	test_numSamples = 0
	dump_dict = {}
	dumpVids_wAttentionMask_startSampleIdxThisBatch = 0
	for ele_idx, loader_ele in enumerate(tqdm(test_loader)):
		if task_type == "classify_oneHot_bestExoPred":
			frames, label, indices, label_multiHot = loader_ele
		else:
			frames, label, indices = loader_ele
		frames = frames.to(device)
		label = label.to(device)
		indices = indices
		if task_type == "classify_oneHot_bestExoPred":
			label_multiHot = label_multiHot.to(device)

		with torch.no_grad():
			if use_relativeCameraPoseLoss:
				feats, feats_pose = vid_encoder(frames)
			else:
				feats = vid_encoder(frames)
			feats = feats.detach()
			out = model(feats)

		if task_type in ["classify_oneHot", "classify_oneHot_bestExoPred"]:
			if len(label.shape) == 2:
				label = label.squeeze(-1)
			loss = F.cross_entropy(out, label,) 
		elif task_type == 'match_dist':
			loss = F.kl_div(F.log_softmax(out, dim=1), F.softmax(label, dim=1), reduction='batchmean')
		elif task_type == "classify_multiHot_bestExoPred":
			loss = F.binary_cross_entropy(torch.sigmoid(out), label,)

		test_loss += loss.item() * len(label)
		if task_type in ["classify_oneHot", "match_dist",]:
			if task_type in ["match_dist"]:
				label = torch.argmax(label, dim=1).long()
			test_acc += int(torch.sum(torch.argmax(out, dim=1).long() == label))
		elif task_type == "classify_oneHot_bestExoPred":
			out_oneHot = torch.zeros((out.shape[0], out.shape[1])).to(out.device)
			out_oneHot[list(range(len(out_oneHot))), torch.argmax(out, dim=1).detach().tolist()] = 1
			test_acc += len(set(torch.where((out_oneHot + label_multiHot) == 2.0)[0].detach().tolist()))
		elif task_type == "classify_multiHot_bestExoPred":
			out_multiHot = (out > 0.5).float()
			test_acc += len(set(torch.where((out_multiHot + label) == 2.0)[0].detach().tolist()))

		# raise ValueError
		assert len(indices) == len(label), print(len(indices), len(label))
		for index in indices:
			index_scalar = index.item() 
			assert index_scalar not in dump_dict
			dump_dict[index_scalar] = out[index_scalar - test_numSamples].tolist()

		test_numSamples += len(label)

	test_loss /= max(test_numSamples, 1)
	test_acc /= max(test_numSamples, 1)
	print(f"Test: loss -- {test_loss:.4f}, accuracy -- {test_acc:.4f}")

	json_dmp(dump_dict, dump_fp)
