from datasets.dataset import train_dataset, val_dataset
from trainer import train_n_val
from common.dist_utils import *
from common.utils import *

import os
import argparse
import warnings, random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def main():
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description = "Lang-View training on LEMMA")

	parser.add_argument("--seed", dest="seed", type=int, default=0, help="Random seed value")
	parser.add_argument("--run-dir", type=str, default="runs/DIRNAME", help="Run directory")
	parser.add_argument("--data-parallel", action="store_true", help="Train w/ DataParallel")
	parser.add_argument("--log-tb", action="store_true", help="Log tensorboard")

	parser.add_argument("--epochs", type=int, default=5000, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=24, help='Batch size')
	parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
	parser.add_argument("--num-trainSamples", type=int, default=12000, help="Number of train samples per epoch")
	parser.add_argument("--num-trainIterations", type=int, default=47, help="Number of distributed train iterations per epoch")
	parser.add_argument("--num-valIterations", type=int, default=4, help="Number of distributed val iterations per epoch")
	parser.add_argument("--num-valSamples", type=int, default=960, help="Number of val samples per epoch")

	parser.add_argument("--optimizer-type", type=str, default="adam_w", help="from ['adam' | 'adam_w'] (default: 'adam_w')")

	parser.add_argument("--isLemma-dataset", action="store_true")
	parser.add_argument("--trainDatapoints-filePath", type=list_of_strs__or__str,
						default="data/lemma/labels/train/videoLlama_cider_all3Agree.pkl",
						help="Path to file with train datapoints")
	parser.add_argument("--valDatapoints-filePath", type=list_of_strs__or__str,
						default="data/lemma/labels/val/videoLlama_cider_all3Agree.pkl",
						help="Path to file with val datapoints")
	parser.add_argument("--multiBestViewAggregator-multiPseudoLabler", action="store_true")

	parser.add_argument("--trainDatapoints-captioner-filePath", type=list_of_strs__or__str,
						default="data/labels/train/videoLlama_cider.pkl")
	parser.add_argument("--valDatapoints-captioner-filePath", type=list_of_strs__or__str,
						default="data/labels/val/videoLlama_cider.pkl")

	parser.add_argument('--datapoint-videoClips-dir', type=str, 
						default='data/lemma/datapoint_images', 
						help='Datapoint video clips dir. Data needs to be downloaded from the original dataset website and '+\
							 'extracted, and "data/lemma/datapoint_images" needs to point to "data-002" sub-directory')
	parser.add_argument("--use-datapointVideoClips", action="store_true")

	parser.add_argument("--task-type", type=str, default='classify_oneHot', help="Task type from ['classify_oneHot', 'match_dist',]")
	parser.add_argument("--randomize-trainLabel-forOneHot", action="store_true",
						help="Randomize training label for one-hot best view pred")
	parser.add_argument("--randomize-trainViewOrder", action="store_true")

	parser.add_argument("--all-views", type=list_of_strs__or__str, default='fpv1,master', help="List of all views")
	parser.add_argument("--num-frames", type=int, default=8, help="Number of frames (default: 8)")
	parser.add_argument("--frame-height", type=int, default=224, help="Frame height (default: 224)")
	parser.add_argument("--frame-width", type=int, default=224, help="Frame width (default: 224)")
	parser.add_argument("--dont-square-frames", action="store_true")
	parser.add_argument("--frame-horizontalFlip", action="store_true",)	# default=[1024, 128]
	parser.add_argument("--frame-colorJitter", type=list_of_floats, default="0.0,0.0,0.0")	# default=[1024, 128]

	parser.add_argument('--unfreeze-videoEncoder', action="store_true")
	parser.add_argument("--videoEncoder-dropout", type=float, default=0.)

	parser.add_argument('--recog-arc', type=str, default="egovlp_v2", help="Recognition architecture from ['egovlp_v2',]")
	parser.add_argument("--vidEncoder-ckptPath", type=none_or_str, 
						default="pretrained_checkpoints/egovlpV2_model_best_egoExo30nov2024.pth",
						help="Path to pretrained video encoder checkpoint")
	parser.add_argument("--use-egovlpV2-patchLevelVisualFeats", action="store_true")
	parser.add_argument("--egovlpV2-patchLevelVisualFeats-convOutDims", type=int, default=384)
	parser.add_argument("--egovlpV2-depth", type=int, default=12,)
	parser.add_argument("--egovlpV2-feedFourFrames", action="store_true")

	parser.add_argument("--egovlpV2-encodeWdinoV2", action="store_true")
	parser.add_argument("--egovlpV2-dinoV2-numRegTokens", type=int, default=0)
	parser.add_argument("--egovlpV2-dinoV2-interpAntiAlias", action="store_true",)
	parser.add_argument("--egovlpV2-dinoV2-interpOffset", type=float, default=0.1)

	parser.add_argument("--use-egoVlpV2-takeVideoFeats-usingStartNendTime", action="store_true")
	parser.add_argument("--use-egoVlpV2-takeVideoFeats-usingCenterTime", action="store_true")
	parser.add_argument("--maxStartNendTimeDiff-use-egoVlpV2-takeVideoFeats", type=int,
						default=2)
	parser.add_argument("--padFeatWithZero-use-egoVlpV2-takeVideoFeats-usingStartNendTime", action="store_true")

	parser.add_argument("--use-transformerPol", action="store_true")
	parser.add_argument("--numLayers-transformerPol", type=int, default=2)
	parser.add_argument("--transformerPol-dropout", type=float, default=0.)
	parser.add_argument("--addPE-transformerPol", action="store_true")
	parser.add_argument("--linearLayer-dims", type=list_of_ints, default="1024,128")	# default=[1024, 128]
	parser.add_argument("--linearLayer-dropout", type=float, default=0.)

	parser.add_argument("--use-minMultiHotLoss", action="store_true")
	parser.add_argument("--use-randMultiHotLoss", action="store_true")
	parser.add_argument("--use-bceMultiHotLoss", action="store_true")
	parser.add_argument("--use-klLoss", action="store_true")

	parser.add_argument("--balanceCLasses-inLoss", action="store_true")

	parser.add_argument("--use-relativeCameraPoseLoss", action="store_true")
	parser.add_argument("--useRelu-relativeCameraPoseLoss", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-poseEncoder-dropout", type=float, default=0.)
	parser.add_argument("--maskOut-invalidRelativeCameraPoseLoss-inTraining", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationOnly", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationInAngles", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationInQuarts", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationAsClasses", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationClassSize", type=float, default=30)
	parser.add_argument("--relativeCameraPoseLoss-coordsNormalized", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsInAngles", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsAsClasses", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsClassSize", type=float, default=30)
	parser.add_argument("--relativeCameraPoseLoss-convOutDims", type=int, default=64)
	parser.add_argument("--relativeCameraPoseLoss-refType", type=str, default="first_view",
						help="from ['first_view' | 'all_views' | 'ego_view']")
	parser.add_argument("--relativeCameraPoseLoss-stopGradientRefPose", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-weight", type=float, default=1.0)
	parser.add_argument("--relativeCameraPoseLoss-frameType", type=str, default="all",
						help="from ['all' | 'center']")
	parser.add_argument("--cameraPose-dir", type=str,
						default="data/ego_exo4d/camera_extrinsics/takeWithCP__2__startNendTimestamp__2__timestamp_n_startNendClipName_n_startNendFrameIdx_n_listAtomicDescriptions__obeyingTakeLenConstraint")
	parser.add_argument("--relativeCameraPoseLoss-lossType", type=str, default="l2",
						help="options from ['l1' | 'l2']")

	parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--lr-videoEncoder", type=float, default=1e-5, help="Learning rate")

	parser.add_argument("--distributed", action="store_true", help="Run distributed")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dist-url", type=str, default="env://")
	parser.add_argument("--world-size", type=int, default=1)

	args = parser.parse_args()

	seed = args.seed 
	if args.distributed:
		init_distributed_mode(args)
		seed = seed + get_rank(args)

	print(args)
	print("-" * 80)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.distributed:
		cudnn.benchmark = False
		cudnn.deterministic = True

	if is_dist_avail_and_initialized(args):
		dist.barrier()
	train_data = train_dataset(args, **vars(args))
	val_data = val_dataset(args, **vars(args))

	train_loader = None
	val_loader = None
	if not args.distributed:
		train_loader = torch.utils.data.DataLoader(train_data,
												   batch_size=args.batch_size,
												   shuffle=True,
												   num_workers=args.num_workers,
												   drop_last=True,
												   )

		val_loader = torch.utils.data.DataLoader(val_data,
												 batch_size=args.batch_size,
												 shuffle=False,
												 num_workers=args.num_workers,	
												 drop_last=True,
												 )	

	if (not os.path.isdir(args.run_dir)) and is_main_process(args):
		os.makedirs(args.run_dir)

	old_tb_dir = os.path.join(args.run_dir, "tb")
	if os.path.isdir(old_tb_dir) and is_main_process(args):
		for old_tb_idx in range(1, 10000):
			if not os.path.isdir(os.path.join(args.run_dir, f"tb_{old_tb_idx}")):
				new_tb_dir = os.path.join(args.run_dir, f"tb_{old_tb_idx}")
				os.system(f"mv {old_tb_dir} {new_tb_dir}")
				break

	writer = None
	if args.log_tb and is_main_process(args):
		writer = SummaryWriter(log_dir=old_tb_dir, flush_secs=30,) 

	train_n_val(train_data if args.distributed else train_loader,
				val_data if args.distributed else val_loader,
				writer,
				args,
				**vars(args))


if __name__ == '__main__':
	main()
