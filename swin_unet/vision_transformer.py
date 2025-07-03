# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

try:
    from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
except ImportError:
    from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, drop_path_rate=0.1, ape=False,
                 patch_norm=True, use_checkpoint=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            patch_size=patch_size,
                                            in_chans=in_chans,
                                            num_classes=num_classes,
                                            embed_dim=embed_dim,
                                            depths=depths,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop_rate=drop_rate,
                                            drop_path_rate=drop_path_rate,
                                            ape=ape,
                                            patch_norm=patch_norm,
                                            use_checkpoint=use_checkpoint)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


if __name__ == '__main__':
    x = torch.randn((2, 3, 1024, 1024))
    net = SwinUnet(img_size=1024, num_classes=2, patch_size=4, window_size=8, in_chans=3, mlp_ratio=4.0)
    out = net(x)
    print(out.shape)
