import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ostrack', choices=['ostrack'],
                        help='training script name')
    # parser.add_argument('--config', type=str, default='FE108_T13', help='yaml configure file name')
    parser.add_argument('--config', type=str, default='FE108_T13', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))

def prepare_input(resolution):
    global device
    x1 = torch.randn((1, *resolution)).to(device)
    x2 = torch.randn((1, 7, *resolution)).to(device)
    x3 = torch.randn((1, 3, 256, 256)).to(device)
    return dict(static_template = x1, dynamic_templates = x2, search= x3)

def evaluate_METrack_V4(model, template, dynamic_template, search, dynamic_template_feature):
    '''Speed Test'''
    # macs1, params1 = profile(model, inputs=(template, dynamic_template, search),
    #                          custom_ops=None, verbose=False)
    # macs, params = clever_format([macs1, params1], "%.3f")
    # flops = clever_format([macs1*2], "%.3f")
    # print('overall macs is ', macs)
    # print('overall FLOPs is ', flops)
    # print('overall params is ', params)


    from ptflops import get_model_complexity_info
    macs2, params2 = get_model_complexity_info(model, (3, 128, 128), 
                    input_constructor=prepare_input,
                        as_strings=True, print_per_layer_stat=True, verbose=True)
    # macs, params = clever_format([macs2*2, params2], "%.3f")
    print('overall FLOPs is ', macs2)
    print('overall params is ', params2)


    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # 转换为百万（M）单位
    total_params_in_million = params / 1e6

    print(f'Total parameters: {total_params_in_million:.2f}M')

    T_w = 500
    T_t = 1000
    # print("testing speed ...")
    # torch.cuda.synchronize()
    # with torch.no_grad():
    #     # overall
    #     for i in range(T_w):
    #         _ = model.backbone.forward_dynamic_features_speed_test(dynamic_template)
    #         _ = model.inference(dynamic_template_feature, template, search)
    #     start = time.time()
    #     for i in range(T_t):
    #         _ = model.backbone.forward_dynamic_features_speed_test(dynamic_template)
    #         _ = model.inference(dynamic_template_feature, template, search)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     avg_lat = (end - start) / T_t
    #     print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    #     print("FPS is %.2f fps" % (1. / avg_lat))



def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    global device
    device = "cuda:0"
    
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "ostrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_ostrack
        model = model_constructor(cfg, training=False)
        print(model.backbone.layers)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)  # (1, 3, 128, 128)
        dynamic_template = torch.randn(bs, 7, 3, z_sz, z_sz)  # (1, 7, 3, 128, 128)
        search = torch.randn(bs, 3, x_sz, x_sz)
        dynamic_template_feature = torch.randn((bs, 448, 384))  # (1, 448, 384)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        dynamic_template = dynamic_template.to(device)
        dynamic_template_feature = dynamic_template_feature.to(device)
        search = search.to(device)

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        evaluate_METrack_V4(model, template, dynamic_template, search, dynamic_template_feature)
        # if merge_layer <= 0:
        #     evaluate_vit(model, template, search)
        # else:
        #     evaluate_vit_separate(model, template, search)

    else:
        raise NotImplementedError
