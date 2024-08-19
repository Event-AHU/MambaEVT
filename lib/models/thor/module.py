import abc
import random
import numpy as np
import torch
import torch.nn.functional as F
import math

class THOR_Wrapper():
    def __init__(self, net, st_capacity, lt_capacity, sample_interval, lb=0.90, response_threshold=0.8):
        self.response_threshold = response_threshold
        self.st_capacity = st_capacity
        self.lt_capacity = lt_capacity
        self._curr_type = 'st'
        self._frame_no = 0
        self.interval = sample_interval
        self.net = net
        self.lb = lb
        self.lt_count = 0
        self.st_count = 0

    def setup(self, rawImage):
        """
        initialize the short-term and long-term module
        """
        self.st_module = ST_Module(template_capacity=self.st_capacity, network=self.net)
        self.st_module.fill(rawImage)
        self.lt_module = LT_Module(template_capacity=self.lt_capacity, network=self.net, lb=self.lb)
        self.lt_module.fill(rawImage)


    def update(self, rawImage):
        """
        update the short-term and long-term module
        """
        
        '''
        相似度方法
                # 使用相似度来作为判断依据，发现最终都会使用LT
        
        '''


        if self._frame_no % self.interval == 0: 
            div_scale = self.st_module.update(rawImage)
            # print('frame no: {}, div_scale: {}'.format(self._frame_no, div_scale))
            self.lt_module.update(rawImage, div_scale=div_scale, time=self._frame_no)
        
        cur_feature = self.net.backbone.forward_dynamic_features_inference([rawImage])
        st_curr_sims = self.st_module.pairwise_similarities(cur_feature)
        lt_curr_sims = self.lt_module.pairwise_similarities(cur_feature)

        if torch.max(st_curr_sims) >= torch.max(lt_curr_sims):
            self._curr_type = 'st'
            self.st_count += 1
            return self.st_module.get_dynamic_template_features()
        else:
            self._curr_type = 'lt'
            self.lt_count += 1
            return self.lt_module.get_dynamic_template_features()

        

class TemplateModule():
    def __init__(self, network, template_capacity,  feat_len_z=64):
        self.is_full = False
        self.template_capacity = template_capacity  # 模板数量
        self.template_raw_list = []
        self.sorted_template_raw_list = []
        self._templates_features_stack = None
        self._templates_features_stack_sim = None
        self._base_sim = 0
        self._gram_matrix = None
        self.feat_len_z = feat_len_z
        self.network = network
        self.timestamp_list = []

    def __len__(self):
        return self.template_capacity

    def pairwise_similarities(self, T_n, to_cpu=False):
        """
        卷积法计算相似度
        calculate similarity of given template to all templates in memory
        将一个(1, N, C)的模板T_n与内存中的所有模板进行比较，返回一个(M,)的相似度向量

        assert isinstance(T_n, torch.Tensor)
        B, N, C = T_n.size()
        H = W = int(math.sqrt(N))
        T_n = T_n.view(1, C, H, W)
        # nums = int(self._templates_features_stack.size()[1]/self.feat_len_z)
        
        # self._templates_features_stack = self._templates_features_stack.view(nums, C, H, W)

        sims = F.conv2d(T_n, self._templates_features_stack_sim)
        sims = sims.flatten()
        return sims
        """
        # 将A的形状从(1, N, C)调整为(N, C)
        T = T_n.clone().squeeze(0)
        
        # 将LIST的形状从(1, N*m, C)调整为(N*m, C)
        LIST = self._templates_features_stack.clone().squeeze(0)
        
        # 初始化sims变量，用于存储每个LIST元素与A的皮尔逊相关性系数
        sims = torch.empty((LIST.size(0) // T.size(0),))
        
        # 计算A的均值和标准差
        mean_T_n = torch.mean(T, dim=0)
        std_T_n = torch.std(T, dim=0)
        
        # 遍历LIST中的每个元素（假设每个元素大小也是(N, C)）
        for i in range(0, LIST.size(0), T.size(0)):
            # 提取LIST中的对应元素
            element = LIST[i:i+T.size(0)]
            
            # 计算元素的均值和标准差
            mean_element = torch.mean(element, dim=0)
            std_element = torch.std(element, dim=0)
            
            # 计算皮尔逊相关性系数
            cov = torch.sum((T - mean_T_n) * (element - mean_element), dim=0)
            pearson_coeff = cov / (std_T_n * std_element)
            
            # 将计算结果赋值给sims
            sims[i // T.size(0)] = torch.mean(pearson_coeff)  # 取平均是为了处理多通道C的情况
        
        return sims

    def _calculate_gram_matrix(self):
        '''
        self._templates_features_stack.shape : (n*64, 384)
        '''
        # 使用PyTorch计算所有模板的成对相似度
        nums = int(self._templates_features_stack.size()[1]/self.feat_len_z)
        dists = []
        for idx in range(nums):
            left = idx*self.feat_len_z
            right = (idx+1)*self.feat_len_z
            dists.append(self.pairwise_similarities(self._templates_features_stack[:,left:right]))
            # dists.append(self.pairwise_similarities(self._templates_features_stack[:,idx*self.feat_len_z:(idx+1)*self.feat_len_z]))

        # dists = [self.pairwise_similarities(self._templates_features_stack[:,idx*self.feat_len_z:(idx+1)*self.feat_len_z]) for idx in range(nums)]
        
        # 使用PyTorch的squeeze函数，并且不需要转换成numpy数组
        gram_matrices = [dists[i].squeeze() for i in range(len(dists))]
        gram_matrices = torch.stack(gram_matrices, dim=0)
        
        return gram_matrices

    def _set_temp(self, temp, idx, time):
        """
        switch out the template at idx
        """
        self.template_raw_list[idx] = temp
        self.timestamp_list[idx] = time


    @abc.abstractmethod
    def update(self, temp):
        """
        check if template should be taken into memory
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fill(self, temp):
        """
        fill all slots in the memory with the given template   # 填充函数
        """
        for _ in range(self.template_capacity):
            self.template_raw_list.append(temp)
            
        self.sorted_template_raw_list = self.template_raw_list
        self._gen_templates_features_stack()

        self._gram_matrix = self._calculate_gram_matrix()
        self._base_sim = self._gram_matrix[0, 0]
        self.timestamp_list = [0] * self.template_capacity


    def _gen_templates_features_stack(self):
        self._templates_features_stack = []
        for temp in self.template_raw_list:
            self._templates_features_stack.append(self.network.backbone.forward_dynamic_features_inference([temp]))
        self._templates_features_stack = torch.cat(self._templates_features_stack, dim=1)

        B, N, C = self._templates_features_stack.size()
        N = N//self.feat_len_z
        H = W = int(math.sqrt(self.feat_len_z))
        self._templates_features_stack_sim = self._templates_features_stack.clone().view(N, C, H, W)

    def get_dynamic_template_features(self):
        return self.network.backbone.forward_dynamic_features_inference(self.template_raw_list)[:, -self.feat_len_z:]
        # return self._templates_features_stack[:, -self.feat_len_z:] 


class ST_Module(TemplateModule):
    def __init__(self, network, template_capacity):
        super(ST_Module, self).__init__(network, template_capacity)
    
    @staticmethod
    def normed_div_measure(matrix):
        """ calculate the normed diversity measure of matirx, the higher the more diverse """
        # 确认输入张量t是方阵
        assert matrix.shape[0] == matrix.shape[1], "Input tensor must be square"
        
        dim = matrix.shape[0] - 1
        triu_no = int(dim * (dim + 1) / 2) # 注意这个 dim=n-1, n为方阵维度，此处计算的是原方阵上三角数目
        
        # 使用PyTorch的triu函数来计算上三角矩阵的和
        triu_sum = torch.sum(torch.triu(matrix, 1))
        
        # 计算归一化多样性度量
        # measure = 1 - triu_sum / (matirx.max() * triu_no)
        measure = triu_sum / (matrix[0, 0] * triu_no) # 多样性计算公式， np.triu(t, 1)提取上三角部分。公式为: 上三角元素和 / 首元素*上三角元素数目
        return measure


    def _update_gram_matrix(self, new_template):
        '''
        new_template.shape -> (B, N, C)
        '''
        # 计算当前的距离
        curr_sims = self.pairwise_similarities(new_template)

        # sorted_indices = torch.argsort(curr_sims, descending=True)

        # # 使用排序索引对self.template_raw_list进行排序
        # self.sorted_template_raw_list = [self.template_raw_list[i] for i in sorted_indices]

        curr_sims = curr_sims.unsqueeze(1)  # 使用 PyTorch 的 unsqueeze 方法

        # 更新 Gram 矩阵
        all_dists_new = torch.cat([
            self._gram_matrix, curr_sims], dim=1)  # 水平拼接
        curr_sims = torch.cat((torch.zeros(1, 1).to(curr_sims.device), curr_sims.T), dim=1)  # 水平拼接
        all_dists_new = torch.cat([
            all_dists_new, curr_sims], dim=0)  # 垂直拼接

        # 删除索引为 0 的行和列，即最旧的模板
        self._gram_matrix = all_dists_new[1:, 1:]


    def update(self, temp):
        """
        append to the current memory and rebuild canvas
        return div_scale (diversity of the current memory)
        """
        # update distance matrix and calculate the div scale
        self.template_raw_list.append(temp)
        self.template_raw_list.pop(0)
        self._gen_templates_features_stack()
        
        temp = self.network.backbone.forward_dynamic_features_inference([temp])
        self._update_gram_matrix(temp)

        # self._templates_features_stack = self.network.backbone.forward_dynamic_features_inference(self.template_raw_list)
        

        return self.normed_div_measure(matrix=self._gram_matrix)


    

class LT_Module(TemplateModule):
    def __init__(self, network, template_capacity, lb=0.883665, lb_type='dynamic'):
        '''
        lb : 0.90, 0.75, 0.60, 0.45
        '''

        super(LT_Module, self).__init__(network, template_capacity)
        self._lb = lb
        self._lb_type = lb_type
        self._filled_idx = 0
        
    # def get_dynamic_template_features(self):
    #     # shuffle_list = self.template_raw_list.copy()
    #     sorted_indices = sorted(range(len(self.timestamp_list)), key=lambda k: self.timestamp_list[k])
    #     sorted_timestamp_list = [self.template_raw_list[i] for i in sorted_indices]
        
    #     return self.network.backbone.forward_dynamic_features_inference(sorted_timestamp_list)[:, -self.feat_len_z:]

    def _throw_away_or_keep(self, curr_sims, self_sim, div_scale):
        """
        determine if we keep the template or not
        if the template is rejected: return -1 (not better) or -2 (rejected by lower bound)
        if we keep the template: return idx where to switch
        """
        base_sim = self._base_sim
        curr_sims = curr_sims.unsqueeze(1)

        # normalize the gram_matrix, otherwise determinants are huge
        gram_matrix_norm = self._gram_matrix / base_sim
        curr_sims_norm = curr_sims / base_sim

        # check if distance to base template is below lower bound
        if self._lb_type == 'static':
            reject = (curr_sims[0] < self._lb * base_sim)

        elif self._lb_type == 'dynamic':
            lb = self._lb - (1 - div_scale)
            lb = torch.clamp(lb, 0.0, 1.0)
            reject = (curr_sims[0] < lb * base_sim)
            # print('Reject: ', reject, 'curr_sims: ', curr_sims[0].item(), 
            #       'lb: ', lb.item(), 'base_sim: ', base_sim.item(), 'lb * base_sim: ', (lb * base_sim).item())

        elif self._lb_type == 'ensemble':
            reject = not torch.all(curr_sims_norm > self._lb)

        else:
            raise TypeError(f"lower boundary type {self._lb_type} not known.")

        if reject:
            return -2

        # fill the memory with adjacent frames if they are not
        # populated with something different than the base frame yet
        if self._filled_idx < (self.template_capacity - 1):
            self._filled_idx += 1
            throwaway_idx = self._filled_idx

        # determine if and in which spot the template increases the current gram determinant
        else:
            curr_det = torch.det(gram_matrix_norm)

            # start at 1 so we never throwaway the base template
            dets = torch.zeros((self.template_capacity - 1,))
            for i in range(self.template_capacity - 1):
                mat = gram_matrix_norm.clone()
                mat[i + 1, :] = curr_sims_norm.T
                mat[:, i + 1] = curr_sims_norm.T
                mat[i + 1, i + 1] = self_sim / base_sim
                dets[i] = torch.det(mat)

            # check if any of the new combinations is better than the prev. gram_matrix
            max_idx = torch.argmax(dets)
            if curr_det > dets[max_idx]:
                throwaway_idx = -1
            else:
                throwaway_idx = max_idx + 1

        assert throwaway_idx != 0
        return throwaway_idx if throwaway_idx != self.template_capacity else -1

    # @staticmethod
    # def save_det(d, p):
    #     if os.path.exists(p):
    #         old_det = np.load(p)
    #     else:
    #         old_det = np.array([])
    #     np.save(p, np.concatenate([old_det, d.reshape(-1)]))

    def _update_gram_matrix(self, curr_sims, self_sim, idx):
        """
        update the current gram matrix
        """
        curr_sims = curr_sims.unsqueeze(1)
        # add the self similarity at throwaway_idx spot
        curr_sims[idx] = self_sim

        self._gram_matrix[idx, :] = curr_sims.T
        self._gram_matrix[:, idx] = curr_sims.T

        # gram_matrix_norm = self._gram_matrix/self._base_sim
        # curr_det = np.linalg.det(gram_matrix_norm)
        # self.save_det(curr_det, 'determinants_dyn.npy')


    def update(self, temp, div_scale, time):
        """
        decide if the templates is taken into the lt module
        """

        # calculate the "throwaway_idx", the spot that the new template will take
        temp_features = self.network.backbone.forward_dynamic_features_inference([temp])
        curr_templatae = temp_features.clone()
        B, N, C = curr_templatae.size()
        H = W = int(math.sqrt(N))
        curr_templatae = curr_templatae.view(1, C, H, W)
        curr_sims = self.pairwise_similarities(temp_features)

        self_sim = F.conv2d(curr_templatae, curr_templatae).squeeze().item()
        throwaway_idx = self._throw_away_or_keep(curr_sims=curr_sims, self_sim=self_sim,
                                                 div_scale=div_scale)

        # if the idx is -2 or -1, the template is rejected, otherwise we update
        if throwaway_idx == -2 or throwaway_idx == -1:
            # print('div_scale: {}, reject: {}'.format(div_scale, 'True'))
            pass # rejected
        else:
            # print('div_scale: {}, reject: {}'.format(div_scale, 'False'))
            self._set_temp(temp=temp, idx=throwaway_idx, time=time)
            # sorted_indices = torch.argsort(curr_sims, descending=True)

            # 使用排序索引对self.template_raw_list进行排序
            # self.sorted_template_raw_list = [self.template_raw_list[i] for i in sorted_indices]

            self._update_gram_matrix(curr_sims=curr_sims, self_sim=self_sim, idx=throwaway_idx)
            self._gen_templates_features_stack()
            # self._templates_features_stack = self.network.backbone.forward_dynamic_features_inference(self.sorted_template_raw_list)
