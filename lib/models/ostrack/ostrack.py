"""
Basic ostrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.ostrack.models_mamba import create_block
from timm.models import create_model
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from lib.models.memnet import MemNet




class OSTrack(nn.Module):
    """ This is the base class for ostrack """

    # def __init__(self, visionmamba, memory, box_head,  aux_loss=False, head_type="CORNER"):
    def __init__(self, visionmamba, box_head,  aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
        self.box_head = box_head
        # self.memory = memory
        # self.mem_cell = mem_cell

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_sz_z = int(box_head.feat_sz/2)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_len_z = int(self.feat_sz_z ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self,
                static_template: torch.Tensor, 
                dynamic_templates: torch.Tensor,
                # template: torch.Tensor, # [bs, 3, 256, 256]
                search: torch.Tensor,         # [bs, 3, 128, 128]
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        dynamic_feature = self.backbone.foward_dynamic_features(dynamic_templates)[:, -self.feat_len_z:]
                           
        x = self.backbone.forward_tripple_features(static_z=static_template, 
                                  dynamic_z =dynamic_feature,
                                  x=search,
                                    inference_params=None, 
                                    if_random_cls_token_position=False, 
                                    if_random_token_rank=False)
        # Forward head
        feat_last = x              # x.shape = torch.Size([2, 320, 384])
        # if isinstance(x, list):
        #     feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        # out.update(aux_dict)
        out['backbone_feat'] = x
        # dynamic_features = out['top60patches']
        return out

    
    def inference(self, dynamic_template: torch.Tensor, static_template: torch.Tensor, search: torch.Tensor):
        # self.dynamic_template = dynamic_template[:, -self.feat_len_z:]
        # self.static_template = static_template
        x = self.backbone.inference(dynamic_template, static_template, search,feat_len_s=self.feat_len_s, 
                                    feat_len_z=self.feat_len_z,)
        # Forward head
        feat_last = x              # x.shape = torch.Size([2, 320, 384])
        out = self.forward_head(feat_last, None)
        out['backbone_feat'] = x
        return out
    

    def forward_head(self, feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        search_feature = feature[:, -self.feat_len_s:]               # search_feature.shape = torch.Size([2, 256, 384])
        opt = (search_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()           # opt.shape = torch.Size([2, 1, 384, 256])
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)                       # opt_feat.shape = torch.Size([2, 384, 16, 16])

        
        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        # 为了提速而省去一切if操作
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
            #    'top60patches': selected_patches,
                }
        return out


def build_ostrack(cfg, training=True):
   
    backbone = create_model(
            model_name='vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
            pretrained='prj_dir/pretrained_models/vim_s_midclstok_80p5acc.pth',
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            img_size=224
            )

    hidden_dim = 384
    box_head = build_box_head(cfg, hidden_dim)
    # mem_cell = MemNet(cfg, is_train=True)

    # model = OSTr
    
    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE, flush=True)

    return model