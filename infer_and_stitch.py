import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2 as cv

from datasets.dataset import test_dataset
from datasets.utils import frame_unnormalize
from models import pol
from common.utils import loadModel_trainer, list_of_ints, list_of_strs__or__str, state_dict_data_parallel_fix


def _filter_state_dict_by_shape(load_sd, curr_sd):
    """Keep only keys whose shapes match the current module."""
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in load_sd.items():
        if k in curr_sd and tuple(v.shape) == tuple(curr_sd[k].shape):
            out[k] = v
    return out


def build_models(args):
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    n_available_gpus = torch.cuda.device_count()

    # Build encoder + classifier
    vid_encoder = pol.videoEncoder(vars(args))
    model = pol.pol_v1(vars(args))

    vid_encoder = vid_encoder.to(device)
    model = model.to(device)
    if args.data_parallel and n_available_gpus > 0:
        vid_encoder = torch.nn.DataParallel(vid_encoder, device_ids=list(range(n_available_gpus)), output_device=0)
        model = torch.nn.DataParallel(model, device_ids=list(range(n_available_gpus)), output_device=0)

    vid_encoder.eval()
    model.eval()

    # Load checkpoint
    ckpt_dir = os.path.join(args.run_dir, "data")
    assert os.path.isdir(ckpt_dir), f"Checkpoint dir not found: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, f"{args.checkpoint_fileName}.pth")
    assert os.path.isfile(ckpt_file), f"Checkpoint not found: {ckpt_file}"
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # Fix potential DataParallel prefixes in checkpoints saved without distributed
    try:
        ckpt['model'] = state_dict_data_parallel_fix(ckpt['model'], model.state_dict())
    except Exception:
        pass
    if args.unfreeze_videoEncoder and 'video_encoder' in ckpt:
        try:
            ckpt['video_encoder'] = state_dict_data_parallel_fix(ckpt['video_encoder'], vid_encoder.state_dict())
        except Exception:
            pass
        # Filter out shape-mismatched keys (e.g., different rel-pose class sizes)
        ckpt['video_encoder'] = _filter_state_dict_by_shape(ckpt['video_encoder'], vid_encoder.state_dict())

    # Also filter model in case of minor head diffs
    ckpt['model'] = _filter_state_dict_by_shape(ckpt['model'], model.state_dict())

    loadModel_trainer(ckpt, model, vid_encoder=vid_encoder if args.unfreeze_videoEncoder else None, kwargs=vars(args), is_test=True)

    return vid_encoder, model, device


def stitch_best_views(args):
    # Build dataset + dataloader
    ds = test_dataset(args, **vars(args))
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=False)

    vid_encoder, model, device = build_models(args)

    # Video writer setup
    out_path = args.output_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fps = args.output_fps
    size = (args.frame_width, args.frame_height)  # (W, H)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(out_path, fourcc, fps, size)

    total_written = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Infer & stitch"):
            # Unpack batch according to task type
            if args.task_type == "classify_oneHot_bestExoPred":
                frames, label, indices, label_multiHot = batch
            else:
                frames, label, indices = batch

            frames = frames.to(device)  # (B, V, T, H, W, C), normalized (egovlp)

            # Forward
            if args.use_relativeCameraPoseLoss:
                feats, _ = vid_encoder(frames)
            else:
                feats = vid_encoder(frames)
            logits = model(feats)

            # Optionally exclude specified views by masking their logits
            if hasattr(args, 'exclude_views') and args.exclude_views is not None:
                exclude_views = args.exclude_views if isinstance(args.exclude_views, list) else [args.exclude_views]
                view_to_idx = {v: i for i, v in enumerate(args.all_views)}
                mask_idx = [view_to_idx[v] for v in exclude_views if v in view_to_idx]
                if len(mask_idx) == len(args.all_views):
                    raise ValueError("All views are excluded; nothing left to select.")
                if len(mask_idx) != len(exclude_views):
                    missing = [v for v in exclude_views if v not in view_to_idx]
                    if missing:
                        print(f"Warning: exclude-views not in all-views and will be ignored: {missing}")
                if mask_idx:
                    logits[:, mask_idx] = float('-inf')

            pred = torch.argmax(logits, dim=1)  # (B,)

            # For each sample in batch, select predicted view frames and write to video
            # frames: (B, V, T, H, W, C) in normalized RGB
            B, V, T, H, W, C = frames.shape
            for b in range(B):
                view_idx = pred[b].item()
                chosen = frames[b, view_idx]  # (T, H, W, C)

                # Unnormalize to uint8 RGB
                chosen_unnorm = frame_unnormalize(chosen, input_frame_norm_type='egovlp_v2')  # (T,H,W,C) uint8
                chosen_np = chosen_unnorm.cpu().numpy()  # uint8

                # Write frames as BGR to VideoWriter
                for t in range(T):
                    rgb = chosen_np[t]
                    if rgb.shape[1] != size[0] or rgb.shape[0] != size[1]:
                        rgb = cv.resize(rgb, size, interpolation=cv.INTER_LINEAR)
                    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
                    writer.write(bgr)
                    total_written += 1

    writer.release()
    print(f"Saved stitched video to: {out_path} ({total_written} frames)")


def main():
    parser = argparse.ArgumentParser(description="Ego-Exo4D: infer best views and stitch video")

    # Core config (mirror test.py defaults where possible)
    parser.add_argument("--run-dir", type=str, default="runs/egoExo4d_release", help="Run directory with checkpoints")
    parser.add_argument("--checkpoint-fileName", type=str, default="valBestCkpt_maxCaptioningScore", help="Checkpoint file name (without .pth)")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true", help="Use DataParallel")
    parser.add_argument("--distributed", action="store_true", help="Placeholder to satisfy checkpoint loader")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--testDatapoints-filePath", type=str, default="data/ego_exo4d/labels/test.pkl", help="Path to test datapoints pkl")
    parser.add_argument('--datapoint-videoClips-dir', type=str, default='data/ego_exo4d/clips', help='Datapoint clips dir')
    parser.add_argument("--use-datapointVideoClips", action="store_true")

    parser.add_argument("--task-type", type=str, default='classify_oneHot', help="'classify_oneHot' or 'match_dist'")
    parser.add_argument("--all-views", type=list_of_strs__or__str, default='aria,1,2,3,4', help="List of all views (e.g., aria,1,2,3,4)")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument("--dont-square-frames", action="store_true")

    # Head (classifier) config — must match training
    parser.add_argument("--linearLayer-dims", dest="linearLayer_dims", type=list_of_ints, default="1024,128")
    parser.add_argument("--linearLayer-dropout", dest="linearLayer_dropout", type=float, default=0.0)

    parser.add_argument('--unfreeze-videoEncoder', action="store_true")
    parser.add_argument("--videoEncoder-dropout", type=float, default=0.)

    parser.add_argument('--recog-arc', type=str, default="egovlp_v2")
    parser.add_argument("--vidEncoder-ckptPath", type=str, default="pretrained_checkpoints/egovlpV2_model_best_egoExo30nov2024.pth")

    # Relative camera pose loss flags (match README test command for Ego-Exo4D)
    parser.add_argument("--use-relativeCameraPoseLoss", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationInAngles", dest="relativeCameraPoseLoss_rotationInAngles", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationAsClasses", dest="relativeCameraPoseLoss_rotationAsClasses", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-coordsInAngles", dest="relativeCameraPoseLoss_coordsInAngles", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-coordsAsClasses", dest="relativeCameraPoseLoss_coordsAsClasses", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationClassSize", dest="relativeCameraPoseLoss_rotationClassSize", type=float, default=30)
    parser.add_argument("--relativeCameraPoseLoss-coordsClassSize", dest="relativeCameraPoseLoss_coordsClassSize", type=float, default=30)

    # Output
    parser.add_argument('--output-path', type=str, default='runs/egoExo4d_release/best_view_stitched.mp4')
    parser.add_argument('--output-fps', type=int, default=8, help='FPS for the stitched video')
    parser.add_argument('--exclude-views', type=list_of_strs__or__str, default=None, help='Comma-separated views to exclude from selection (e.g., aria or 1,4)')

    args = parser.parse_args()

    # Sanity checks
    assert isinstance(args.all_views, list), "all_views must be a list like 'aria,1,2,3,4'"
    assert os.path.isfile(args.testDatapoints_filePath), f"Not found: {args.testDatapoints_filePath}"
    assert os.path.isdir(args.datapoint_videoClips_dir), f"Not found: {args.datapoint_videoClips_dir}"
    assert os.path.isdir(args.run_dir), f"Not found: {args.run_dir}"

    # Force datapoint clips usage (Ego-Exo4D released data format)
    args.use_datapointVideoClips = True

    stitch_best_views(args)


if __name__ == '__main__':
    main()
