from datasets.dataset import test_dataset
from trainer import test
from common.dist_utils import *
from common.utils import *

import os
import argparse
import warnings, random
import numpy as np

import torch


def main():
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description = "Lang-View testing on Ego-Exo4D")

	parser.add_argument("--seed", dest="seed", type=int, default=0, help="Random seed value")
	parser.add_argument("--run-dir", type=str, default="runs/DIRNAME", help="Run directory")
	parser.add_argument("--checkpoint-fileName", type=str, default="valBestCkpt_maxCaptioningScore")
	parser.add_argument("--data-parallel", action="store_true", help="Test w/ DataParallel")
	# 本当のデフォルトは64
	parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
	parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")

	parser.add_argument("--testDatapoints-filePath", type=str, 
						default="data/ego_exo4d/labels/test.pkl",	
						help="Path to file with train datapoints")

	parser.add_argument('--datapoint-videoClips-dir', type=str, 
						default='data/ego_exo4d/clips', 
						help='Datapoint video clips dir')
	parser.add_argument("--use-datapointVideoClips", action="store_true")

	parser.add_argument("--task-type", type=str, default='classify_oneHot', help="Task type from ['classify_oneHot', 'match_dist',]")

	parser.add_argument("--all-views", type=list_of_strs__or__str, default='aria,1,2,3,4', help="List of all views")
	parser.add_argument("--num-frames", type=int, default=8, help="Number of frames (default: 8)")
	parser.add_argument("--frame-height", type=int, default=224, help="Frame height (default: 224)")
	parser.add_argument("--frame-width", type=int, default=224, help="Frame width (default: 224)")
	parser.add_argument("--dont-square-frames", action="store_true")

	parser.add_argument('--unfreeze-videoEncoder', action="store_true")
	parser.add_argument("--videoEncoder-dropout", type=float, default=0.)

	parser.add_argument('--recog-arc', type=str, default="egovlp_v2", help="Recognition architecture from ['egovlp_v2',]")
	parser.add_argument("--vidEncoder-ckptPath", type=str, 
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

	parser.add_argument("--use-relativeCameraPoseLoss", action="store_true")
	parser.add_argument("--useRelu-relativeCameraPoseLoss", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-poseEncoder-dropout", type=float, default=0.)
	parser.add_argument("--relativeCameraPoseLoss-rotationOnly", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationInAngles", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationInQuarts", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationAsClasses", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-rotationClassSize", type=float, default=30)
	parser.add_argument("--relativeCameraPoseLoss-coordsInAngles", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsNormalized", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsAsClasses", action="store_true")
	parser.add_argument("--relativeCameraPoseLoss-coordsClassSize", type=float, default=30)
	parser.add_argument("--relativeCameraPoseLoss-convOutDims", type=int, default=64)

	parser.add_argument("--distributed", action="store_true", help="Run distributed")

	args = parser.parse_args()
	print(args)
	print("-" * 80)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	test_data = test_dataset(args, **vars(args))
	test_loader = torch.utils.data.DataLoader(test_data,
											 batch_size=args.batch_size,
											 shuffle=False,
											 num_workers=args.num_workers,	
											 drop_last=False,
											 )	

	assert os.path.isdir(args.run_dir)

	x = test(test_loader, **vars(args))
	# print(x)


if __name__ == '__main__':
	main()
