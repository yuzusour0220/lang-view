import time
import numpy as np

import torch
from torch.utils.data import DataLoader

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

KINETICS_DEFAULT_MEAN = (0.45, 0.45, 0.45)
KINETICS_DEFAULT_STD = (0.225, 0.225, 0.225)

LAVILA_EPIC_DEFAULT_MEAN = (0.42481294, 0.4578275 , 0.40821073)   # (108.3272985, 116.7460125, 104.09373615000001), (0.42481294, 0.4578275 , 0.40821073)
LAVILA_EPIC_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)     # (68.5005327, 66.6321579, 70.32316305), (0.26862954, 0.26130258, 0.27577711)

EGOVLP_V2_DEFAULT_MEAN = (0.485, 0.456, 0.406)
EGOVLP_V2_DEFAULT_STD = (0.229, 0.224, 0.225)


def frame_normalize(tensor, input_frame_norm_type="egovlp_v2", dont_scale=False, return_meanNstd=False):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if input_frame_norm_type == "kinetics":
        mean = KINETICS_DEFAULT_MEAN
        std = KINETICS_DEFAULT_STD
    elif input_frame_norm_type == "lavila_epic":
        mean = LAVILA_EPIC_DEFAULT_MEAN
        std = LAVILA_EPIC_DEFAULT_STD
    elif input_frame_norm_type == "egovlp_v2":
        # print('h2')
        mean = EGOVLP_V2_DEFAULT_MEAN
        std = EGOVLP_V2_DEFAULT_STD
    else:
        raise ValueError

    if return_meanNstd:
        return mean, std

    if dont_scale:
        assert tensor.dtype == torch.float32
    else:
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        elif tensor.dtype == torch.float32:
            pass
        else:
            raise ValueError

    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)

    tensor = (tensor - mean) / (std + 1e-8)

    return tensor


def frame_unnormalize(tensor, input_frame_norm_type="egovlp_v2", ):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if input_frame_norm_type == "kinetics":
        mean = KINETICS_DEFAULT_MEAN
        std = KINETICS_DEFAULT_STD
    elif input_frame_norm_type == "lavila_epic":
        mean = LAVILA_EPIC_DEFAULT_MEAN
        std = LAVILA_EPIC_DEFAULT_STD
    elif input_frame_norm_type == "egovlp_v2":
        mean = EGOVLP_V2_DEFAULT_MEAN
        std = EGOVLP_V2_DEFAULT_STD
    else:
        raise ValueError

    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)

    tensor_unnorm = torch.clip(((tensor * (std + 1e-8)) + mean) * 255, min=0, max=255).byte()

    return tensor_unnorm


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


class PrefetchLoader(object):
    """
    Modified from https://github.com/ChenRocks/UNITER.

    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch

            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)
