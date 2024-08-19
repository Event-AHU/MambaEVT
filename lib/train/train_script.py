import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ostrack import build_ostrack
# forward propagation related
from lib.train.actors import OSTrackActor
# for import modules
import importlib
from thop import profile
from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "ostrack":
        net = build_ostrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    # if torch.cuda.device_count() > 1:
    #     print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    #     net = nn.DataParallel(net)

    '''
    参数统计
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = net.to(device)
    # # 计算总参数量
    # import numpy as np
    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

    # # 转换为百万（M）单位
    # total_params_in_million = params / 1e6
    # print(f'Total parameters: {total_params_in_million:.2f}M')
    
    # input1 = torch.randn(1,3, 128, 128).to(device)
    # input2 = torch.randn(1,2, 3, 128, 128).to(device)
    # input3 = torch.randn(1, 3, 256, 256).to(device)
    # flops, params = profile(net, inputs=(input1,input2,input3))
    
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("params=", str(params/1e6)+'{}'.format("M"))
    # print('----'*20)
    # from calflops import calculate_flops
    # flops, macs, params = calculate_flops(model=net, 
    #                                   input_shape=((1,3, 128, 128), (1,2, 3, 128, 128), (1, 3, 256, 256)),
    #                                   output_as_string=True,
    #                                   output_precision=4)
    # print('flops, macs, params', flops, macs, params)
    # return
    # net.cuda()
    
    
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)