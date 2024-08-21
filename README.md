MambaEVT: Event Stream based Visual Object Tracking using State Space Model

# :dart: Abstract 
Event camera-based visual tracking has drawn more and more attention in recent years due to the unique imaging principle and advantages of low energy consumption, high dynamic range, and dense temporal resolution. Current event-based tracking algorithms are gradually hitting their performance bottlenecks, due to the utilization of vision Transformer and the static template for target object localization. In this paper, we propose a novel Mamba-based visual tracking framework that adopts the state space model with linear complexity as a backbone network. The search regions and target template are fed into the vision Mamba network for simultaneous feature extraction and interaction. The output tokens of search regions will be fed into the tracking head for target localization. More importantly, we consider introducing a dynamic template update strategy into the tracking framework using the Memory Mamba network. By considering the diversity of samples in the target template library and making appropriate adjustments to the template memory module, a more effective dynamic template can be integrated. The effective combination of dynamic and static templates allows our Mamba-based tracking algorithm to achieve a good balance between accuracy and computational cost on multiple large-scale datasets, including EventVOT, VisEvent, and FE240hz. The source code of this work will be released upon acceptance.

# :video_camera: Demo Video





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
[[OSTrack]()] 
[[Mamba]()] 
[[FE108 Dataset]()] 
[[EventVOT Dataset]()] 
[[VisEvent Dataset]()] 

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




















