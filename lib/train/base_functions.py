import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import EventVOT, VisEvent, Fe108
# from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, EventVOT 
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        # assert name in ["EventVOT", "EventVOT_VAL","COESOT","FE240","Felt","VisEvent"]
        if name == "EventVOT":
            datasets.append(EventVOT(settings.env.eventvot_dir, image_loader=image_loader))
        if name == "EventVOT_VAL":
            datasets.append(EventVOT(settings.env.eventvot_val_dir, image_loader=image_loader))
        if name == "Fe108":
            datasets.append(Fe108(settings.env.fe108_dir, image_loader=image_loader))
        if name == "Fe108_VAL":
            datasets.append(Fe108(settings.env.fe108_val_dir, image_loader=image_loader))
        if name == "VisEvent":
             datasets.append(VisEvent(settings.env.visevent_dir, image_loader=image_loader))
        if name == "VisEvent_VAL":
             datasets.append(VisEvent(settings.env.visevent_val_dir, image_loader=image_loader))
        
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, dynamic_nums = cfg.TEST.DYNAMIC_NUMS)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler,
                               pin_memory=False)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls, dynamic_nums = cfg.TEST.DYNAMIC_NUMS)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL, pin_memory=False)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
