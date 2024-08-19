import math

from lib.models.ostrack import build_ostrack
from lib.models.thor import THOR_Wrapper

from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import random
import time
import torch.nn.functional as F
import numpy as np
# from .utils import to_numpy, print_color

'''
        曾经使用的置信度measure
        # print('----'*40)
        # print(response.flatten().sort(descending=True)[0][:10])

        # top1response = response.max()
        # top1response = 0 if top1response < self.params.cfg.TEST.THRESHOLD else top1response

        # self.threshold = update_threshold(self.threshold, response.flatten().sort(descending=True)[0][:5])
        # print(self.threshold)

        # typical_confidence = get_bbox_confidence(*(pred_boxes.mean(dim=0)*self.params.search_size), response[0][0])  # 代表性置信度
        # self.threshold = self.params.cfg.TEST.alpha * self.threshold + (1 - self.params.cfg.TEST.alpha) * typical_confidence
        # typical_confidence = 0 if typical_confidence < self.threshold else typical_confidence
        '''


class ModelManager:
    _instance = None
    _model = None

    def __new__(cls, cfg, training=True):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._model = build_ostrack(cfg, training)
        return cls._instance

    def get_model(self):
        return self._model

def update_threshold(current_threshold, confidence_values, k=5, alpha=0.95):
    weights = torch.exp(torch.linspace(-1., 0., k))[::-1]  # 指数加权并逆序
    weighted_confidences = confidence_values[:k] * weights
    average_confidence = torch.sum(weighted_confidences) / torch.sum(weights)
    new_threshold = alpha * current_threshold + (1 - alpha) * average_confidence
    return new_threshold

def get_bbox_confidence(cx, cy, w, h, confidence_map):
    # 计算 bbox 的边界
    x_min = cx - w / 2
    x_max = cx + w / 2
    y_min = cy - h / 2
    y_max = cy + h / 2

    # 映射到特征图坐标
    x_min_idx = torch.clamp(torch.floor(x_min/16).int(), min=0, max=15)
    x_max_idx = torch.clamp(torch.ceil(x_max/16).int(), min=0, max=15)
    y_min_idx = torch.clamp(torch.floor(y_min/16).int(), min=0, max=15)
    y_max_idx = torch.clamp(torch.ceil(y_max/16).int(), min=0, max=15)

    # 提取覆盖区域的置信度值
    covered_confidences = confidence_map[y_min_idx:y_max_idx+1, x_min_idx:x_max_idx+1]

    # 计算最终置信度
    bbox_confidence = torch.max(covered_confidences)

    return bbox_confidence

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        self.LONGTERM_LIBRARY_NUMS = params.cfg.TEST.LONGTERM_LIBRARY_NUMS
        self.SHORTTERM_LIBRARY_NUMS = params.cfg.TEST.SHORTTERM_LIBRARY_NUMS
        # net, st_capacity, lt_capacity, sample_interval

        
        # self.DYNAMIC_NUMS = self.params.cfg.TEST.DYNAMIC_NUMS
        # self.threshold = .5
        
        model_manager = ModelManager(params.cfg, training=False)    
        network = model_manager.get_model()

        # network = build_ostrack(params.cfg, training=False)
        
        # state_dict = torch.load(params.checkpoint, map_location='cpu')
        # for key in list(state_dict['net'].keys()):
        #     if 'total_' in key:
        #         del state_dict['net'][key]
        # network.load_state_dict(state_dict['net'], strict=True)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_list = []

        self.thor_wrapper = THOR_Wrapper(net=self.network, st_capacity=params.cfg.TEST.SHORTTERM_LIBRARY_NUMS,
                                    lt_capacity=params.cfg.TEST.LONGTERM_LIBRARY_NUMS, 
                                    sample_interval=params.cfg.TEST.SAMPLE_INTERVAL, lb=params.cfg.TEST.THOR_LOWER_BOUND)

    def initialize(self, image, start_frame_idx, info: dict):
        self.frame_no = 0
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)  # template.tensors -> shape:(3, 128, 128)
        with torch.no_grad():
            self.static_z = template.tensors
            self.z_dict1 = template
            self.z_rawimg_list = []
            self.z_rawimg_list.append(template.tensors)
            self.stacked_z = template.tensors
            self.stacked_z_features = self.network.backbone.foward_dynamic_features(template.tensors.unsqueeze(0))
            self.normed_div_measure_value = 0.0
            # template.tensors -> shape:(1, 3, 128, 128)
            self.thor_wrapper.setup(template.tensors)

            
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = start_frame_idx
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, feat_z_len=64):
        self.info = info
        self.frame_no += 1
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            if len(info['previous_output']) == 0:   # 根据有序字典长度判断是否是第一帧
                dynamic_template = self.thor_wrapper.update(self.stacked_z)
            else:
                dynamic_template = self.thor_wrapper.update(info['previous_output']['prediction'])
            out_dict = self.network.inference(
                dynamic_template=dynamic_template, 
                static_template=self.static_z, search=x_dict.tensors)
        
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # extract the tracking result   image -> H,W,C
        tracking_result_arr, resize_factor, tracking_result_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        prediction = self.preprocessor.process(tracking_result_arr, tracking_result_amask_arr)

        top1response = response.max()

        # self.getCAM2(response, image, self.frame_id)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,
                    # "prediction": prediction.tensors}
                    "prediction": prediction.tensors,
                    "response": response}
        
        

    def getCAM2(features, img, idx):      

        save_path =  'prj/save/'                                                                    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # os.path = /home/tcm/PycharmProjects/siamft/pysot_toolkit/trackers
        features = features.to("cpu")
        features = features.squeeze(1).detach().numpy()
        img = cv2.resize(img, (256, 256))
        img = img
        img = np.array(img, dtype=np.uint8)
        # mask = features.sum(dim=0, keepdims=False)
        mask = features
        # mask = mask.detach().cpu().numpy()
        mask = mask.transpose((1, 2, 0))
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = cv2.resize(mask, (256,256))
        mask = 255 * mask
        mask = mask.astype(np.uint8)
        heatmap = cv2.applyColorMap(255-mask, cv2.COLORMAP_JET)

        img = cv2.addWeighted(src1=img, alpha=0.6, src2=heatmap, beta=0.4, gamma=0)
        name = '/attn_%d.png' % idx
        cv2.imwrite(save_path + name, img)
        # cv2.imwrite(name, img)


    def pltshow(self, pred_map, name):
        import matplotlib.pyplot as plt
        plt.figure(2)
        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        pred_name = os.path.dirname(__file__) + '/response/' + str(name) + '.png'
        plt.savefig(pred_name, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(100)





        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2

        # # 提取分数图
        # score_map = out_dict['score_map']
        # score_map = score_map.squeeze().cpu().numpy()  # 转换为 (16, 16)

        # # 使用 matplotlib 生成热力图
        # plt.imshow(score_map, cmap='jet', interpolation='bicubic')  # 使用 'bicubic' 插值
        # plt.axis('off')

        # # 保存热力图到文件
        # plt.savefig('heatmap.png', bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.close()

        # # 读取热力图
        # heatmap = cv2.imread('heatmap.png', cv2.IMREAD_UNCHANGED)  # 使用透明背景
        # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 调整大小为 (W, H)

        # # 提取 alpha 通道作为遮罩
        # alpha_channel = heatmap[:, :, 3] / 255.0  # 归一化到 [0, 1]

        # # 复制 event image 以便融合
        # fused_image = image.copy()

        # # 使用简单加权平均方法进行融合
        # fused_image = (heatmap[:, :, :3] // 2 + image // 2).astype(np.uint8)

        # # 保存结果到文件
        # cv2.imwrite('heatmap_overlay.jpg', fused_image)

        # heatmap[:,:,:3]//2 + image//2
        # self.activation_map()

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,
                    # "prediction": prediction.tensors}
                    "prediction": prediction.tensors}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights



def get_tracker_class():
    return OSTrack


