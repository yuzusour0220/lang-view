import json
import os
import pickle
import numpy as np
from collections import OrderedDict
import torch


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        raise ValueError
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def ospif(file):
    return os.path.isfile(file)


def ospid(dir_):
    return os.path.isdir(dir_)


def pkl_dmp(obj, fp):
    with open(fp, "wb") as fo:
        pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)


def pkl_ld(fp):
    with open(fp, "rb") as fi:
        pkl_content = pickle.load(fi)
    return pkl_content


def json_ld(fp):
    with open(fp, "r") as fi:
        json_content = json.load(fi)
    return json_content


def json_dmp(obj, fp, indent=None):
    with open(fp, "w") as fo:
        if indent is None:
            json.dump(obj, fo)
        else:
            assert isinstance(indent, int)
            json.dump(obj, fo, indent=indent)


def list_of_ints(arg):
    if arg == 'None':
        return []
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    if arg == 'None':
        return []
    return list(map(float, arg.split(',')))


def list_of_strs__or__str(arg):
    if len(arg.split(',')) == 1:
        return str(arg.split(',')[0])
    else:
        return [str(ele) for ele in arg.split(',')]


def none_or_str(value):
    if value == 'None':
        return None
    return value


def saveModel_trainer(kwargs,
                      ckpt_dir,
                      epoch,
                      model,
                      optimizer,
                      best_metric,
                      video_encoder=None,
                      is_best=False,
                      best_loss=float('inf'),
                      is_bestLoss=False,
                      best_captioningScores=[float('-inf')] * 1,
                      is_bestCaptioningScores=[False] * 1,
                      task_type="classify_oneHot"):
    checkpoint_paths = [os.path.join(ckpt_dir, 'valLastCkpt.pth')]

    egoVlpV2_vis2textSim_labler = kwargs["egoVlpV2_vis2textSim_labler"] if ("egoVlpV2_vis2textSim_labler" in kwargs) else False
    if is_best:    
        checkpoint_paths.append(os.path.join(ckpt_dir, 'valBestCkpt.pth'))
    if ("bestExoPred" in kwargs["task_type"]) and is_bestLoss:
        checkpoint_paths.append(os.path.join(ckpt_dir, 'valBestCkpt_minLoss.pth'))

    for captioner_idx, is_bestCaptioningScore in enumerate(is_bestCaptioningScores):
        if is_bestCaptioningScore and (captioner_idx == 0):
            checkpoint_paths.append(os.path.join(ckpt_dir, f'valBestCkpt_maxCaptioningScore.pth'))

    for checkpoint_path in checkpoint_paths:
        ckpt_dct = {
            'model': model.module.state_dict() if kwargs["distributed"] else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': kwargs,
            'epoch': epoch,
        }
        if video_encoder is not None:
            ckpt_dct["video_encoder"] = video_encoder.module.state_dict() if kwargs["distributed"] else video_encoder.state_dict()
        ckpt_dct['max_acc'] = best_metric
        ckpt_dct['min_loss'] = best_loss

        ckpt_dct['max_captioningScores'] = best_captioningScores


        torch.save(ckpt_dct, checkpoint_path)


def loadModel_trainer(checkpoint,
                      model,
                      vid_encoder=None,
                      optimizer=None,
                      kwargs=None,
                      is_test=False):
    task_type = kwargs["task_type"]
    is_vidEncoderLoading_strict = True
    if is_test:
        is_vidEncoderLoading_strict = not (kwargs["use_relativeCameraPoseLoss"] if ("use_relativeCameraPoseLoss" in kwargs) else False)    
    if kwargs["distributed"]:
        model.module.load_state_dict(checkpoint['model'])
        if vid_encoder is not None:
            vid_encoder.module.load_state_dict(checkpoint['video_encoder'], strict=is_vidEncoderLoading_strict)
    else:
        model.load_state_dict(checkpoint['model'])
        if vid_encoder is not None:
            vid_encoder.load_state_dict(checkpoint['video_encoder'], strict=is_vidEncoderLoading_strict)

    if not is_test:
        if 'optimizer' in checkpoint:
            assert optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])

        min_loss = float('inf')
        max_acc = float('-inf')
        max_captioningScores = [float('-inf')] * 1
        max_acc = checkpoint['max_acc']
        if 'min_loss' in checkpoint:
            min_loss = checkpoint['min_loss']
            if "max_captioningScores" in checkpoint:
                max_captioningScores = checkpoint["max_captioningScores"]
                best_metric = (max_acc, max_captioningScores, min_loss)
            else:
                best_metric = (max_acc, min_loss)
        else:
            if "max_captioningScores" in checkpoint:
                raise NotImplementedError
            best_metric = max_acc 

        epoch = 0
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            epoch += 1

        return best_metric, epoch


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict
