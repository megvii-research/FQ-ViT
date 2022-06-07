# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def load_weights_from_npz(model,
                          url,
                          check_hash=False,
                          progress=False,
                          prefix=''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    def _get_cache_dir(child_dir=''):
        """
        Returns the location of the directory where models are cached (and creates it if necessary).
        """
        hub_dir = torch.hub.get_dir()
        child_dir = () if not child_dir else (child_dir, )
        model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _download_cached_file(url, check_hash=True, progress=False):
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(_get_cache_dir(), filename)
        if not os.path.exists(cached_file):
            hash_prefix = None
            if check_hash:
                r = torch.hub.HASH_REGEX.search(
                    filename)  # r is Optional[Match[str]]
                hash_prefix = r.group(1) if r else None
            torch.hub.download_url_to_file(url,
                                           cached_file,
                                           hash_prefix,
                                           progress=progress)
        return cached_file

    def adapt_input_conv(in_chans, conv_weight):
        conv_type = conv_weight.dtype
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv_weight = conv_weight.float()
        O, I, J, K = conv_weight.shape
        if in_chans == 1:
            if I > 3:
                assert conv_weight.shape[1] % 3 == 0
                # For models with space2depth stems
                conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
                conv_weight = conv_weight.sum(dim=2, keepdim=False)
            else:
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
        elif in_chans != 3:
            if I != 3:
                raise NotImplementedError(
                    'Weight format not supported by conversion.')
            else:
                # NOTE this strategy should be better than random init, but there could be other combinations of
                # the original RGB input layer weights that'd work better for specific cases.
                repeat = int(math.ceil(in_chans / 3))
                conv_weight = conv_weight.repeat(1, repeat, 1,
                                                 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(in_chans))
        conv_weight = conv_weight.to(conv_type)
        return conv_weight

    def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]
        if num_tokens:
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[
                0, num_tokens:]
            ntok_new -= num_tokens
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        gs_old = int(math.sqrt(len(posemb_grid)))
        if not len(gs_new):  # backwards compatibility
            gs_new = [int(math.sqrt(ntok_new))] * 2
        assert len(gs_new) >= 2
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                          -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid,
                                    size=gs_new,
                                    mode='bicubic',
                                    align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3,
                                          1).reshape(1, gs_new[0] * gs_new[1],
                                                     -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    cached_file = _download_cached_file(url,
                                        check_hash=check_hash,
                                        progress=progress)

    w = np.load(cached_file)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(stem.conv.weight.shape[1],
                             _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(
                            _n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(
                            _n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(
                            _n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1],
                                        _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'],
                       t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1),
            model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(
            model.head, nn.Linear
    ) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None),
                  nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(
            torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T
                for n in ('query', 'key', 'value')
            ]))
        block.attn.qkv.bias.copy_(
            torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1)
                for n in ('query', 'key', 'value')
            ]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
