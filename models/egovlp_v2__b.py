import os
from abc import abstractmethod

import timm
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn

from common.utils import state_dict_data_parallel_fix

from models import roberta
from models.roberta import RobertaModel, _prepare_decoder_attention_mask
from models import heads
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist




class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



with open('/checkpoint/sagnikmjr2002/code/av_bvs/avBvs_policy_v1/EgoNCE_MLM_ITM_Config.yml') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

NUM_FUSE_BLOCK = config_yaml['num_fuse_block']
DIM_TEXT = 768

def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F == self.num_frames, print(F, self.num_frames)
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        return x


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random', dim_text=None, norm_layer=nn.LayerNorm, space_attn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        if dim_text is not None and space_attn:
            self.qkv_text_i2t = nn.Linear(dim_text, dim * 2, bias=qkv_bias)
            self.qkv_i2t = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop_i2t = nn.Dropout(attn_drop)
            self.proj_i2t = nn.Linear(dim, dim)
            self.proj_drop_i2t = nn.Dropout(proj_drop)
            self.alpha_i2t = nn.Parameter(torch.Tensor([0]))
            self.norm_i2t_i = norm_layer(dim)

    def forward(self, x, einops_from, einops_to, y=None, y_mask=None, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q*self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)

        if y is not None:
            B_, N, C = x.shape
            B_text, N_text, C_text = y.shape

            kv_text = (
                self.qkv_text_i2t(y)
                .reshape(B_text, N_text, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            k_text, v_text = kv_text[0], kv_text[1]

            q_i2t = self.qkv_i2t(self.norm_i2t_i(x))
            q_i2t = q_i2t.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_i2t = q_i2t[0]

            # image to text attention
            text_scale = k_text.size(-1) ** -0.5
            q_i2t = q_i2t * text_scale
            attn_i2t = q_i2t @ k_text.transpose(-2, -1)  # B_, nH, N, N_text

            # add image to text bias and text_mask
            if y_mask is not None:
                mask_and_i2t_bias = y_mask.view(B_text, 1, 1, N_text)
                attn_i2t = attn_i2t + mask_and_i2t_bias

            attn_i2t = self.softmax(attn_i2t)
            attn_i2t = self.attn_drop_i2t(attn_i2t)
            y = (attn_i2t @ v_text).transpose(1, 2).reshape(B_, N, C)
            y = self.proj_i2t(y)
            y = self.proj_drop_i2t(y)
            x = x + self.alpha_i2t * y

        return x


class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', dim_text=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, dim_text=dim_text, 
            norm_layer=norm_layer, space_attn=True)

        self.timeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            initialize=time_init, dim_text=dim_text, norm_layer=norm_layer, space_attn=False)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style

    def forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
                time_n, space_f, y=None, y_mask=None):

        time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, n=time_n, y=None, y_mask=None)
        time_residual = x + time_output
        space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                 einops_to_space, f=space_f, y=y, y_mask=y_mask)
        if self.attention_style == 'frozen-in-time':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        return x


class SpaceTimeTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650

    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].

    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', norm_layer=nn.LayerNorm, dim_text=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        print("######USING ATTENTION STYLE: ", attention_style)
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames)
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=attention_style, dim_text=None if i < 6 else DIM_TEXT)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        ## einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        b, curr_frames, channels, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for blk in self.blocks:
            if config_yaml['use_checkpoint']:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, time_n=n, space_f=f)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(blk), x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time)
            else:
                x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time, time_n=n, space_f=f)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




with open('/checkpoint/sagnikmjr2002/code/av_bvs/avBvs_policy_v1/EgoNCE_MLM_ITM_Config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class FrozenInTime(BaseModel):
    def __init__(self,
                 num_frames=8,
                 video_params={"model": "SpaceTimeTransformer",
                                "arch_config": "base_patch16_224",
                                "num_frames": 4,
                                "pretrained": True,
                                 "time_init": "zeros"},
                 text_params={"model": "roberta-base",
                                "pretrained": True,
                                "input": "text"},
                 projection_dim=4096,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='bilinear',
                 config = config,
                 task_names = 'EgoNCE_MLM_ITM',
                 norm_layer = None,
                 embed_dim=768):
        super().__init__()
        video_params["num_frames"] = num_frames
        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.config = config
        self.task_names = task_names
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        if self.text_params['model'].startswith('roberta'):
            self.text_model = RobertaModel.from_pretrained("roberta-base")
        else:
            self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            self.num_frames = video_params['num_frames']
            time_init = 'zeros'
            attention_style = 'frozen-in-time'
            arch_config = 'base_patch16_224'
            vit_init = 'imagenet-21k'
            if arch_config == 'base_patch16_224':
                # vit_model = torch.load("jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                model = SpaceTimeTransformer(num_frames=self.num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
           
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint['model_state'], model.state_dict())
                missing_keys, unexpected_keys = model.load_state_dict(new_vit_dict, strict=False)
                print("egovlp_v2 1 -- missing keys, unexpected keys: ", missing_keys, unexpected_keys)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':

            txt_proj = nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim, bias=False),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            )

        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if ('MLM' in self.task_names or 'ITM' in self.task_names):
            # for FIBER-like cross-attention

            bert_config = RobertaConfig(
                vocab_size=self.config["vocab_size"],
                hidden_size=self.config["hidden_size"],
                num_hidden_layers=self.config["num_layers"],
                num_attention_heads=self.config["num_heads"],
                intermediate_size=self.config["hidden_size"] * self.config["mlp_ratio"],
                #max_position_embeddings=maxlen, [was used in BTGOT script]
                hidden_dropout_prob=self.config["drop_rate"],
                attention_probs_dropout_prob=self.config["drop_rate"],
            )

            self.num_fuse_block=self.config["num_fuse_block"]
            self.num_text_layer=self.config["num_layers"]
            roberta.NUM_FUSE_BLOCK = self.video_model.NUM_FUSE_BLOCK=self.num_fuse_block
            roberta.DIM_IMG=self.config["input_image_embed_size"]
            self.video_model.DIM_TXT=self.config["input_text_embed_size"]

            self.cross_modal_text_transform = nn.Linear(self.config["input_text_embed_size"], self.config["hidden_size"])
            self.cross_modal_text_transform.apply(init_weights)
            self.cross_modal_video_transform = nn.Linear(self.config["input_image_embed_size"], self.config["hidden_size"])
            self.cross_modal_video_transform.apply(init_weights)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.num_patches = self.video_model.patch_embed.num_patches
            self.patches_per_frame = self.num_patches//self.num_frames
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
            self.pre_logits = nn.Identity()


            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.cross_modal_video_pooler = heads.Pooler(self.config["hidden_size"])
            self.cross_modal_video_pooler.apply(init_weights)
            self.cross_modal_text_pooler = heads.Pooler(self.config["hidden_size"])
            self.cross_modal_text_pooler.apply(init_weights)

            ## einops transformations
            self.einops_from_space = 'b (f n) d'
            self.einops_to_space = '(b f) n d'
            self.einops_from_time = 'b (f n) d'
            self.einops_to_time = '(b n) f d'

        if 'MLM' in self.task_names:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if 'ITM' in self.task_names:
            self.itm_score = heads.ITMHead(self.config["hidden_size"] * 2)
            self.itm_score.apply(init_weights)

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            print("egovlp_v2 2 -- missing keys, unexpected keys: ", missing_keys, unexpected_keys)

    def set_device(self, device):
        self.device = device

    # def infer(self, data, video_only=False, return_embeds=True, task_names=None, ret={}):
        
    # def forward(self, data, video_only=False, return_embeds=True, task_names=None, ret={}):

    #     text_data = data['text']
    #     video_data = data['video']
    def forward(self, video_data, text_data, video_only=False, return_embeds=True, task_names=None, ret=None):
        assert ret is None

        # text_data = data['text']
        # video_data = data['video']

        if task_names is not None:
            self.task_names = task_names


        if 'EgoNCE' in self.task_names: 

            text_embeddings = self.compute_text(text_data)
            video_embeddings = self.compute_video(video_data)


            # if return_embeds:
            #     ret.update({'text_embeds':text_embeddings,
            #     'video_embeds':video_embeddings
            #     })

            return video_embeddings, text_embeddings

        if 'ITM' in self.task_names:
            raise ValueError

            b, curr_frames, channels, _, _ = video_data.shape
            video_data_itm = self.video_model.patch_embed(video_data)
            video_data_itm = video_data_itm.flatten(2).transpose(2, 1)
            video_data_itm = video_data_itm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            BF = video_data_itm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_itm = torch.cat((cls_tokens, video_data_itm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_itm.shape[1]
            video_data_itm = video_data_itm + total_pos_embed[:, :curr_patches]
            video_data_itm = self.video_model.pos_drop(video_data_itm)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            

            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    video_data_itm = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_itm = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
                            
            
            text_embeds = self.text_model.embeddings(input_ids=text_data['input_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]


            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if self.config['use_checkpoint']:
                    

                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_itm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_itm), last_norm=True)[0]
                video_data_itm = fuse_video_data

            
            #print("Shape of model output", video_data.shape)
            video_data_itm = self.norm(video_data_itm)[:, 0]
            video_data_itm = self.pre_logits(video_data_itm)

            text_embeds = text_embeds[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)
            video_embeds = self.cross_modal_video_transform(video_data_itm)

            cls_feats_text = self.cross_modal_text_pooler(text_embeds)
            
            cls_feats_video = self.cross_modal_video_pooler(video_embeds)

            cls_feats = torch.cat([cls_feats_text, cls_feats_video], dim=-1)

            ret.update({
                "cross_attn_itm_logits": self.itm_score(cls_feats)
            })


        if 'MLM' in self.task_names:
            raise ValueError

            b, curr_frames, channels, _, _ = video_data.shape
            video_data_mlm = self.video_model.patch_embed(video_data)
            video_data_mlm = video_data_mlm.flatten(2).transpose(2, 1)
            video_data_mlm = video_data_mlm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            BF = video_data_mlm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_mlm = torch.cat((cls_tokens, video_data_mlm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            #print("total_pos_embed shape: ", total_pos_embed.shape)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_mlm.shape[1]
            video_data_mlm = video_data_mlm + total_pos_embed[:, :curr_patches]
            video_data_mlm = self.video_model.pos_drop(video_data_mlm)

            #print("video_data_mlm shape: ", video_data_mlm.shape)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            
            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    video_data_mlm = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_mlm = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
                       
            
            text_embeds = self.text_model.embeddings(input_ids=data['text_mlm_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)

            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]

            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if self.config['use_checkpoint']:

                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                            n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_mlm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_mlm), last_norm=True)[0]
                video_data_mlm = fuse_video_data


            text_embeds = text_embeds #[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)

            ret.update({
                "cross_attn_mlm_logits": self.mlm_score(text_embeds)
            })

        return ret

    
    # def forward(self, data, n_embeds, v_embeds, allgather, n_gpu, args, config, loss_egonce, gpu, return_embeds=True, task_names='EgoNCE_ITM_MLM'):

    #     ret = {}
    #     loss_dict = {}

    #     if 'Feature_Extraction' in task_names:
    #         video_embeddings = self.compute_video(data['video'])
    #         return video_embeddings


    #     if 'EgoNCE' in task_names:

    #         ret = self.infer(data, task_names='EgoNCE')
    #         video_embeds = ret['video_embeds']
    #         text_embeds = ret['text_embeds']
    #         video_embeds = allgather(video_embeds, n_gpu, args)
    #         text_embeds = allgather(text_embeds, n_gpu, args)
    #         n_embeds = allgather(n_embeds, n_gpu, args)
    #         v_embeds = allgather(v_embeds, n_gpu, args)
    #         output = sim_matrix(text_embeds, video_embeds)

    #         if config['loss']['type'] == 'EgoNCE':
    #             sim_v = sim_matrix(v_embeds, v_embeds)
    #             sim_n = sim_matrix(n_embeds, n_embeds)
    #             loss, mask_bool, temp = loss_egonce(output, sim_v, sim_n)
    #         else:
    #             loss, mask_bool, temp = loss_egonce(output)

    #         ret.update({"sim_v2t": output, "sim_t2v": output.t(),})

    #         loss_dict.update({'EgoNCE': loss})

        
    #     # MLM
    #     if 'MLM' in task_names:

    #         ret = self.infer(data, task_names='MLM', ret=ret)

    #         mlm_logits = ret["cross_attn_mlm_logits"].view(-1, 50265)
    #         mlm_labels = data["text_mlm_labels"].view(-1)

    #         mlm_logits = allgather(mlm_logits, n_gpu, args)
    #         mlm_labels = allgather(mlm_labels, n_gpu, args)

    #         loss_mlm = torch.nn.functional.cross_entropy(
    #                             mlm_logits,
    #                             mlm_labels,
    #                             ignore_index=-100,
    #                             ).mean()

    #         loss = loss + loss_mlm

    #         loss_dict.update({"loss_mlm": loss_mlm})


    #     # ITM
    #     if 'ITM' in task_names:
    #         raise ValueError
    #         rank = dist.get_rank()

    #         all_video = allgather(data['video'], n_gpu, args)
    #         all_text_ids = allgather(data['text']['input_ids'], n_gpu, args)
    #         all_text_masks = allgather(data['text']['attention_mask'], n_gpu, args)

    #         pos_len = data['video'].size(0) // 2
    #         neg_len = data['video'].size(0) - pos_len
    #         itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).cuda(gpu, non_blocking=True)

    #         itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    #         batch_size = len(itm_labels)

    #         with torch.no_grad():
    #             weights_v2t = F.softmax(ret['sim_v2t'][batch_size*rank : batch_size * (rank + 1), :]/temp, dim=1)
    #             weights_t2v = F.softmax(ret['sim_t2v'][batch_size*rank : batch_size * (rank + 1), :]/temp, dim=1)

    #             weights_v2t.masked_fill_(mask_bool[batch_size*rank : batch_size * (rank + 1), :], 0)
    #             weights_t2v.masked_fill_(mask_bool[batch_size*rank : batch_size * (rank + 1), :], 0)

    #         data_itm = copy.deepcopy(data)

    #         for idx in range(len(itm_labels)):
    #             if itm_labels[idx] == 1:
    #                 data_itm['video'][idx, :] = all_video[rank*batch_size + idx, :]
    #                 data_itm['text']['input_ids'][idx, :] = all_text_ids[rank*batch_size + idx, :]
    #                 data_itm['text']['attention_mask'][idx, :] = all_text_masks[rank*batch_size + idx, :]


    #             else:
    #                 if np.random.rand() > 0.5:
    #                     neg_idx = torch.multinomial(weights_t2v[idx] + 1e-9, 1).item()
    #                     data_itm['video'][idx, :] = all_video[neg_idx, :]
    #                     data_itm['text']['input_ids'][idx, :] = all_text_ids[rank*batch_size + idx, :]
    #                     data_itm['text']['attention_mask'][idx, :] = all_text_masks[rank*batch_size + idx, :]
    #                 else:
    #                     neg_idx = torch.multinomial(weights_v2t[idx] + 1e-9, 1).item()
    #                     data_itm['video'][idx, :] = all_video[rank*batch_size + idx, :]
    #                     data_itm['text']['input_ids'][idx, :] = all_text_ids[neg_idx, :]
    #                     data_itm['text']['attention_mask'][idx, :] = all_text_masks[neg_idx, :]


    #         ret = self.infer(data_itm, task_names='ITM', ret=ret)

    #         itm_logits = ret["cross_attn_itm_logits"]

    #         itm_logits = allgather(itm_logits, n_gpu, args)
    #         itm_labels = allgather(itm_labels, n_gpu, args)

    #         loss_itm = torch.nn.functional.cross_entropy(itm_logits, itm_labels.long()).mean()

    #         loss = loss + 2*loss_itm

    #         #print("ITM loss: ", loss_itm)
    #         loss_dict.update({"loss_itm": loss_itm})

    #     loss_dict.update({"loss_total": loss})

    #     return loss, loss_dict, ret



    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        if self.config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        if self.config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        if self.config['use_checkpoint']:
            video_embeddings = torch.utils.checkpoint.checkpoint(self.vid_proj, video_embeddings)
        else:
            video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_batch_val(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


if __name__ == "__main__":
    pass
