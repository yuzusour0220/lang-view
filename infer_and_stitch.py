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
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in load_sd.items():
        if k in curr_sd and tuple(v.shape) == tuple(curr_sd[k].shape):
            out[k] = v
    return out


def _inspect_ckpt_for_views_and_task(args):
    """Read checkpoint to discover training-time all_views and task_type.
    Optionally override runtime args to match ckpt for consistent heads.
    """
    ckpt_dir = os.path.join(args.run_dir, "data")
    ckpt_file = os.path.join(ckpt_dir, f"{args.checkpoint_fileName}.pth")
    assert os.path.isfile(ckpt_file), f"Checkpoint not found: {ckpt_file}"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}

    ckpt_views = ckpt_args.get('all_views', None)
    ckpt_task = ckpt_args.get('task_type', None)

    if getattr(args, 'use_views_from_ckpt', False) and ckpt_views is not None:
        if ckpt_views != args.all_views:
            print(f"Info: overriding all_views from ckpt: {ckpt_views} (was {args.all_views})")
            args.all_views = ckpt_views
    else:
        if ckpt_views is not None and ckpt_views != args.all_views:
            print(f"Warning: all_views mismatch. ckpt={ckpt_views} vs runtime={args.all_views}")

    if getattr(args, 'use_task_from_ckpt', False) and ckpt_task is not None:
        if ckpt_task != args.task_type:
            print(f"Info: overriding task_type from ckpt: {ckpt_task} (was {args.task_type})")
            args.task_type = ckpt_task
    else:
        if ckpt_task is not None and ckpt_task != args.task_type:
            print(f"Warning: task_type mismatch. ckpt={ckpt_task} vs runtime={args.task_type}")

    # Try to align important model hyperparams with ckpt to maximize weight compatibility
    align_keys = [
        'use_transformerPol',
        'numLayers_transformerPol',
        'transformerPol_dropout',
        'addPE_transformerPol',
        'linearLayer_dims',
        'linearLayer_dropout',
        'egovlpV2_depth',
        'egovlpV2_feedFourFrames',
        'videoEncoder_dropout',
    ]
    for k in align_keys:
        if k in ckpt_args and hasattr(args, k):
            if getattr(args, k) != ckpt_args[k]:
                print(f"Info: overriding {k} from ckpt: {ckpt_args[k]} (was {getattr(args, k)})")
                setattr(args, k, ckpt_args[k])

    return ckpt


def _summarize_weight_loading(ckpt_model_sd, model):
    try:
        tgt_sd = model.state_dict()
        common = [k for k in ckpt_model_sd.keys() if k in tgt_sd and tuple(ckpt_model_sd[k].shape) == tuple(tgt_sd[k].shape)]
        print(f"Info: weight-compat keys matched: {len(common)}/{len(tgt_sd)}")
        # Classifier summary
        ckpt_cls = [(k, tuple(v.shape)) for k, v in ckpt_model_sd.items() if k.startswith('classifier.') and (k.endswith('.weight') or k.endswith('.bias'))]
        tgt_cls = [(k, tuple(v.shape)) for k, v in tgt_sd.items() if k.startswith('classifier.') and (k.endswith('.weight') or k.endswith('.bias'))]
        if ckpt_cls and tgt_cls:
            print("Info: ckpt classifier layers:")
            for k, s in ckpt_cls:
                print(f"  - {k}: {s}")
            print("Info: runtime classifier layers:")
            for k, s in tgt_cls:
                print(f"  - {k}: {s}")
    except Exception:
        pass


def build_models(args):
    # Inspect ckpt first to align views/task if requested
    ckpt = _inspect_ckpt_for_views_and_task(args)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    n_available_gpus = torch.cuda.device_count()

    vid_encoder = pol.videoEncoder(vars(args))
    model = pol.pol_v1(vars(args))

    vid_encoder = vid_encoder.to(device)
    model = model.to(device)
    if args.data_parallel and n_available_gpus > 0:
        vid_encoder = torch.nn.DataParallel(vid_encoder, device_ids=list(range(n_available_gpus)), output_device=0)
        model = torch.nn.DataParallel(model, device_ids=list(range(n_available_gpus)), output_device=0)

    vid_encoder.eval()
    model.eval()

    # Fix DP prefixes and filter shapes
    try:
        raw_model_sd = ckpt['model']
        ckpt['model'] = state_dict_data_parallel_fix(raw_model_sd, model.state_dict())
    except Exception:
        pass
    # If ckpt contains a fine-tuned video encoder, prefer loading it for inference
    if 'video_encoder' in ckpt:
        try:
            ckpt['video_encoder'] = state_dict_data_parallel_fix(ckpt['video_encoder'], vid_encoder.state_dict())
        except Exception:
            pass
        ckpt['video_encoder'] = _filter_state_dict_by_shape(ckpt['video_encoder'], vid_encoder.state_dict())

    # Warn if classifier out_features mismatches runtime views
    try:
        last_cls_key = None
        last_idx = -1
        for k in ckpt['model'].keys():
            if 'classifier.' in k and k.endswith('.weight'):
                try:
                    idx = int(k.split('classifier.')[1].split('.')[0])
                except Exception:
                    continue
                if idx > last_idx:
                    last_idx = idx
                    last_cls_key = k
        if last_cls_key is not None:
            ckpt_num_classes = ckpt['model'][last_cls_key].shape[0]
            if ckpt_num_classes != len(args.all_views):
                print(f"Warning: classifier out_features mismatch. ckpt={ckpt_num_classes} vs runtime={len(args.all_views)}. Final head will remain randomly initialized.")
    except Exception:
        pass

    ckpt['model'] = _filter_state_dict_by_shape(ckpt['model'], model.state_dict())
    _summarize_weight_loading(ckpt['model'], model)

    loadModel_trainer(ckpt, model, vid_encoder=vid_encoder if 'video_encoder' in ckpt else None, kwargs=vars(args), is_test=True)

    # Friendly logs
    try:
        if 'video_encoder' in ckpt:
            print("Info: loaded fine-tuned video encoder weights from checkpoint.")
        else:
            print("Info: checkpoint has no video_encoder; using pretrained backbone only.")
    except Exception:
        pass

    return vid_encoder, model, device


def _select_best_view_indices(logits: torch.Tensor, all_views):
    """Return LongTensor[B] of best-view indices from logits.
    Handles cases where logits may include extra temporal/spatial dims.
    Strategy: softmax along view-dim, average over other dims, argmax.
    """
    assert logits.dim() >= 2, f"Unexpected logits dim: {logits.shape}"
    B = logits.shape[0]
    num_views = len(all_views)

    # Find class/view dimension by matching size
    view_dim = None
    for d in range(1, logits.dim()):
        if logits.shape[d] == num_views:
            view_dim = d
            break
    if view_dim is None:
        # Fallback to dim=1
        view_dim = 1

    # Move class dim to last and flatten other non-batch dims except class
    x = torch.movedim(logits, view_dim, -1)  # [B, ..., V]
    flat = x.reshape(B, -1, num_views)      # [B, N, V]
    probs = F.softmax(flat, dim=-1)
    mean_probs = probs.mean(dim=1)          # [B, V]
    return torch.argmax(mean_probs, dim=1)


def _compute_mean_probs_over_views(logits: torch.Tensor, all_views):
    """Return probs[B, V] by applying softmax over the view dimension and
    averaging over any extra non-view dims (e.g., temporal tokens).
    """
    assert logits.dim() >= 2, f"Unexpected logits dim: {logits.shape}"
    B = logits.shape[0]
    num_views = len(all_views)

    view_dim = None
    for d in range(1, logits.dim()):
        if logits.shape[d] == num_views:
            view_dim = d
            break
    if view_dim is None:
        view_dim = 1

    x = torch.movedim(logits, view_dim, -1)  # [B, ..., V]
    flat = x.reshape(B, -1, num_views)      # [B, N, V]
    probs = F.softmax(flat, dim=-1)
    mean_probs = probs.mean(dim=1)          # [B, V]
    return mean_probs


def stitch_best_views(args):
    # Align runtime args with ckpt BEFORE building dataset to preserve view order
    _inspect_ckpt_for_views_and_task(args)

    ds = test_dataset(args, **vars(args))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    vid_encoder, model, device = build_models(args)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    fps = args.output_fps
    size = (args.frame_width, args.frame_height)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    class_hist = np.zeros(len(args.all_views), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc="Infer & stitch")):
            if args.task_type == "classify_oneHot_bestExoPred":
                frames, label, indices, label_multiHot = batch
            else:
                frames, label, indices = batch

            frames = frames.to(device)

            if args.use_relativeCameraPoseLoss:
                feats, _ = vid_encoder(frames)
            else:
                feats = vid_encoder(frames)
            logits = model(feats)  # shape may be [B, V] or include extra dims

            # Exclude views if requested
            if getattr(args, 'exclude_views', None):
                exclude = args.exclude_views if isinstance(args.exclude_views, list) else [args.exclude_views]
                v2i = {v: i for i, v in enumerate(args.all_views)}
                mask_idx = [v2i[v] for v in exclude if v in v2i]
                if mask_idx:
                    logits[:, mask_idx] = float('-inf')

            pred_view = _select_best_view_indices(logits, args.all_views)  # (B,)

            if getattr(args, 'debug', False):
                # Compute probs aggregated over non-view dims for robust display
                mean_probs = _compute_mean_probs_over_views(logits, args.all_views)
                top1 = pred_view.cpu().numpy().tolist()
                unique, counts = np.unique(top1, return_counts=True)
                print(f"Debug batch {batch_idx}: top1 -> {list(zip(unique.tolist(), counts.tolist()))}")
                # Show per-sample probabilities for the first N samples
                show_n = min(args.debug_prob_samples, mean_probs.shape[0]) if hasattr(args, 'debug_prob_samples') else min(2, mean_probs.shape[0])
                for b in range(show_n):
                    probs_b = mean_probs[b].detach().cpu().numpy().tolist()
                    probs_fmt = ", ".join([f"{args.all_views[i]}: {probs_b[i]:.3f}" for i in range(len(args.all_views))])
                    print(f"Debug batch {batch_idx} sample {b}: probs -> [{probs_fmt}]")
                # Input stats per view
                Bv, Vv, Tv, Hv, Wv, Cv = frames.shape
                means = frames.mean(dim=(2,3,4,5)).cpu().numpy()  # (B,V)
                vars_ = frames.var(dim=(2,3,4,5)).cpu().numpy()   # (B,V)
                print(f"Debug batch {batch_idx}: per-view means (first 2): {means[:2]}")
                print(f"Debug batch {batch_idx}: per-view vars  (first 2): {vars_[:2]}")

            B, V, T, H, W, C = frames.shape
            for b in range(B):
                view_idx = pred_view[b].item()
                class_hist[view_idx] += 1
                chosen = frames[b, view_idx]  # (T,H,W,C), normalized RGB
                rgb_uint8 = frame_unnormalize(chosen, input_frame_norm_type='egovlp_v2').cpu().numpy()

                # Build filename — use dataset indices if available
                base = str(indices[b].item()) if torch.is_tensor(indices[b]) else f"{batch_idx:05d}_{b:02d}"
                out_path = os.path.join(out_dir, f"{base}_view{args.all_views[view_idx]}.mp4")
                writer = cv.VideoWriter(out_path, fourcc, fps, size)

                for t in range(T):
                    rgb = rgb_uint8[t]
                    if rgb.shape[1] != size[0] or rgb.shape[0] != size[1]:
                        rgb = cv.resize(rgb, size, interpolation=cv.INTER_LINEAR)
                    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
                    writer.write(bgr)
                writer.release()

    # Summary
    total = class_hist.sum()
    if total > 0:
        dist = {args.all_views[i]: int(class_hist[i]) for i in range(len(args.all_views))}
        print(f"Prediction distribution over views: {dist}")
    print(f"Saved stitched videos to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Infer best view per clip and stitch to MP4")

    # Core
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with checkpoints")
    parser.add_argument("--checkpoint-fileName", type=str, required=True, help="Checkpoint file name without .pth")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # Data
    parser.add_argument("--testDatapoints-filePath", type=str, required=True)
    parser.add_argument('--datapoint-videoClips-dir', type=str, required=True)
    parser.add_argument("--use-datapointVideoClips", action="store_true")

    # Task/model
    parser.add_argument("--task-type", type=str, default='classify_oneHot')
    parser.add_argument("--all-views", type=list_of_strs__or__str, default='aria,1,2,3,4')
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument("--dont-square-frames", action="store_true")

    parser.add_argument("--linearLayer-dims", dest="linearLayer_dims", type=list_of_ints, default="1024,128")
    parser.add_argument("--linearLayer-dropout", dest="linearLayer_dropout", type=float, default=0.0)
    parser.add_argument("--use-transformerPol", dest="use_transformerPol", action="store_true")
    parser.add_argument("--numLayers-transformerPol", dest="numLayers_transformerPol", type=int, default=2)
    parser.add_argument("--transformerPol-dropout", dest="transformerPol_dropout", type=float, default=0.0)
    parser.add_argument("--addPE-transformerPol", dest="addPE_transformerPol", action="store_true")

    parser.add_argument('--unfreeze-videoEncoder', action="store_true")
    parser.add_argument("--videoEncoder-dropout", type=float, default=0.)

    parser.add_argument('--recog-arc', type=str, default="egovlp_v2")
    parser.add_argument("--vidEncoder-ckptPath", type=str, default="pretrained_checkpoints/egovlpV2_model_best_egoExo30nov2024.pth")
    parser.add_argument("--egovlpV2-depth", type=int, default=12)
    parser.add_argument("--egovlpV2-feedFourFrames", action="store_true")

    # Rel-pose flags (kept for compatibility)
    parser.add_argument("--use-relativeCameraPoseLoss", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationInAngles", dest="relativeCameraPoseLoss_rotationInAngles", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationAsClasses", dest="relativeCameraPoseLoss_rotationAsClasses", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-coordsInAngles", dest="relativeCameraPoseLoss_coordsInAngles", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-coordsAsClasses", dest="relativeCameraPoseLoss_coordsAsClasses", action="store_true")
    parser.add_argument("--relativeCameraPoseLoss-rotationClassSize", dest="relativeCameraPoseLoss_rotationClassSize", type=float, default=30)
    parser.add_argument("--relativeCameraPoseLoss-coordsClassSize", dest="relativeCameraPoseLoss_coordsClassSize", type=float, default=30)

    # Output
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save per-clip MP4s')
    parser.add_argument('--output-fps', type=int, default=8)
    parser.add_argument('--exclude-views', type=list_of_strs__or__str, default=None)
    parser.add_argument('--use-views-from-ckpt', action='store_true', default=True, help='Override --all-views from checkpoint')
    parser.add_argument('--use-task-from-ckpt', action='store_true', help='Override --task-type from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Print per-batch distribution and input means')
    parser.add_argument('--debug-prob-samples', type=int, default=2, help='How many samples per batch to print full per-view probabilities for')

    args = parser.parse_args()

    assert isinstance(args.all_views, list), "all_views must be a list (e.g., aria,1,2,3,4)"
    args.use_datapointVideoClips = True

    stitch_best_views(args)


if __name__ == '__main__':
    main()
