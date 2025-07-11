
<p align="center">
<img src="https://github.com/Event-AHU/MambaEVT/blob/main/EventVOT_demo.gif" width="600">
</p>


#### [arXiv:2408.10487] MambaEVT: Event Stream based Visual Object Tracking using State Space Model, 
Xiao Wang, Chao wang, Shiao Wang, Xixi Wang, Zhicheng Zhao, Lin Zhu, Bo Jiang 
[[Paper](https://www.arxiv.org/pdf/2408.10487)] 


# :dart: News 
* [2025.07.03] MambaEVT is accepted by **IEEE TCSVT 2025**! 


# :dart: Abstract 
Event camera-based visual tracking has drawn more and more attention in recent years due to the unique imaging principle and advantages of low energy consumption, high dynamic range, and dense temporal resolution. Current event-based tracking algorithms are gradually hitting their performance bottlenecks, due to the utilization of vision Transformer and the static template for target object localization. In this paper, we propose a novel Mamba-based visual tracking framework that adopts the state space model with linear complexity as a backbone network. The search regions and target template are fed into the vision Mamba network for simultaneous feature extraction and interaction. The output tokens of search regions will be fed into the tracking head for target localization. More importantly, we consider introducing a dynamic template update strategy into the tracking framework using the Memory Mamba network. By considering the diversity of samples in the target template library and making appropriate adjustments to the template memory module, a more effective dynamic template can be integrated. The effective combination of dynamic and static templates allows our Mamba-based tracking algorithm to achieve a good balance between accuracy and computational cost on multiple large-scale datasets, including EventVOT, VisEvent, and FE240hz. The source code of this work will be released upon acceptance.

# [![YouTube](https://badges.aleen42.com/src/youtube.svg)](#) Demo Video





----


# :hammer: Environment 


Install env
```
conda env create -f MambaEVT.yaml
conda activate MambaEVT
```

we follow the way of [Vim](https://github.com/hustvl/Vim) to install Mamba packeges. 

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

We've created a demo dataset in `demo_dataset/`

Download pre-trained [Vim-S](https://huggingface.co/hustvl/Vim-small-midclstok) and put it under `$/pretrained_models` then change the path in `lib/models/ostrack/ostrack.py, Line 124`

## Train & Test
```
# train
python tracking/train.py --script ostrack --config EventVOT_MambaEVT_demo --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py ostrack EventVOT_MambaEVT_demo --dataset eventvot --threads 1 --num_gpus 1
```

------

### Test Speed

*Note:* The speeds reported in our paper were tested on a single RTX 3090 GPU. You can find it in `$/tracking/profile_model.py`




### Acknowledgement 
We would like to thank the following works 
[[OSTrack](https://github.com/botaoye/OSTrack)] 
[[Vision Mamba](https://github.com/hustvl/Vim)]  
[[THOR](https://github.com/xl-sr/THOR)]
[[FE108 Dataset](https://zhangjiqing.com/dataset/)] 
[[EventVOT Dataset](https://github.com/Event-AHU/EventVOT_Benchmark)] 
[[VisEvent Dataset](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)] 



## :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@misc{wang2024MambaEVT,
      title={MambaEVT: Event Stream based Visual Object Tracking using State Space Model}, 
      author={Xiao Wang and Chao wang and Shiao Wang and Xixi Wang and Zhicheng Zhao and Lin Zhu and Bo Jiang},
      year={2024},
      eprint={2408.10487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10487}, 
}
```

If you have any questions about these works, please feel free to leave an issue. 




















