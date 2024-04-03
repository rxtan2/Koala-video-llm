import argparse
import os
import sys
import urllib.request
from collections import OrderedDict
import pickle
import time
import logging

import math
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import decord

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import koala.tasks as tasks
from koala.common.config import Config
from koala.common.dist_utils import get_rank, init_distributed_mode
from koala.common.registry import registry

# imports modules for registration
from koala.datasets.builders import *
from koala.models import *
from koala.processors import *
from koala.runners import *
from koala.tasks import *

from koala.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
    get_cosine_schedule_with_warmup,
)

# import datasets
from video_agg_dataloader import *

# tensorboard
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('train VideoGPT')
parser.add_argument('--data_dir', type=str, default="/research/reuben/procedural_video_understanding/data/howto100m/", metavar='FC',
                    help='path to output caption dir')
parser.add_argument('--video_dir', type=str, default="/research/reuben/combined_all_tasks_howto100m_videos_subsampled_frames", metavar='VD',
                    help='path to video dir')
parser.add_argument('--saved_checkpoints_dir', type=str, default="./all_training_weights/", metavar='VD',
                    help='path to video dir')
parser.add_argument('--tensorboard_path', type=str, default="./tensorboard_train_plots", metavar='VD',
                    help='path to video dir')
parser.add_argument('--cfg-path', type=str, default="./train_configs/video_aggregation_finetune.yaml", metavar='CP',
                    help='path to model config file')
parser.add_argument('--options', nargs="+", metavar='CP',
                    help="override some settings in the used config, the key-value pair "
                    "in xxx=yyy format will be merged into config file (deprecate), "
                    "change to --cfg-options instead.")

# distributed training settings
parser.add_argument('--gpu', type=int, default=None, metavar='GPU',
                        help='GPU id to use')
parser.add_argument('--dist_url', type=str, default="tcp://localhost:23456", metavar='DU',
                    help='local host url for distributed training') 
parser.add_argument('--dist_backend', type=str, default="nccl", metavar='DB',
                    help='distributed backend')
parser.add_argument('--num_gpus', type=int, default=1, metavar='NGPU',
                        help='specify number of gpus to use in total')
parser.add_argument('--rank', type=int, default=0, metavar='rank',
                        help='node rank for distributed training')
parser.add_argument('--world_size', type=int, default=1, metavar='WS',
                        help='number of nodes for distributed training')
parser.add_argument('--pin_memory', type=bool, default=False, metavar='PM',
                    help='use pinned memory during training')
parser.add_argument('--resume_training_checkpoint', type=str, default=None, metavar='RTC',
                    help='path to checkpoint to resume training from')

# training configuration
parser.add_argument('--scaling_amp', type=bool, default=True, metavar='SA',
                    help='flag to use mixed precision training')
parser.add_argument('--lr_sched', type=str, default="linear_warmup_cosine_lr", metavar='LRS',
                    help='schedule for annealing learning rate')
parser.add_argument('--num_epochs', type=int, default=2, metavar='NE',
                    help='number of finetuning epochs')
parser.add_argument('--batch_size', type=int, default=2, metavar='BS',
                    help='batch size for training')
parser.add_argument('--num_workers', type=int, default=4, metavar='NW',
                    help='number of worker threads')
parser.add_argument('--init_lr', type=float, default=1e-5, metavar='ILR',
                    help='initial starting learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5, metavar='MLR',
                    help='minimum starting learning rate')
parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='WLR',
                    help='learning rate warmup factor')
parser.add_argument('--accum_grad_iters', type=int, default=1, metavar='AGT',
                    help='number of steps to accumulate the gradient over')
parser.add_argument('--warmup_steps', type=int, default=410, metavar='WS',
                    help='number of warmup steps')
parser.add_argument('--weight_decay', type=float, default=0.05, metavar='WD',
                    help='weight decay')
parser.add_argument('--display_iter', type=int, default=10, metavar='DI',
                    help='log training progress every display_iter steps')
parser.add_argument('--save_checkpoint_iters', type=int, default=100, metavar='SCI',
                    help='Number of steps to save checkpoint within each epoch')

# model configurations
parser.add_argument('--num_frames_per_video', type=int, default=32, metavar='NC',
                    help='number of clips and captions to sample from each video')
parser.add_argument('--num_frames_per_clip', type=int, default=16, metavar='NPPC',
                    help='specify how frames to use per clip')
parser.add_argument('--num_segments', type=int, default=4, metavar='NS',
                    help='specify number of video segments')
parser.add_argument('--hierarchical_agg_function', type=str, default="without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned", metavar='HAF',
                    help='specify function to merge global and clip visual representations')
parser.add_argument('--freeze_llama_proj', type=bool, default=True, metavar='FLP',
                    help='flag to freeze llama linear projection layers during prompt finetuning')
parser.add_argument('--global_region_embed_weight', type=float, default=1e-3, metavar='GREW',
                    help='weight decay')

parser.add_argument('--frame_size', type=int, default=224, metavar='FS',
                    help='size of input frames')

parser.add_argument('--single_feature_per_frame', type=bool, default=True, metavar='SFPF',
                    help='flag to use a single feature per frame to reduce computational demands')
parser.add_argument('--merge_frames', type=bool, default=False, metavar='MF',
                    help='flag to merge multiple frames in pretrained and frozen q-former')
parser.add_argument('--clip_level_generation', type=bool, default=False, metavar='CLG',
                    help='flag to break video into shorter clips')
parser.add_argument('--use_memory', type=bool, default=False, metavar='UM',
                    help='flag to use memory aggregation')
parser.add_argument('--num_memory_tokens', type=int, default=32, metavar='MF',
                    help='flag to use memory aggregation')

parser.add_argument('--first_stage_random_clips', type=bool, default=False, metavar='FSRC',
                    help='flag to vary the number of input frames during training.')
parser.add_argument('--second_stage_random_clips', type=bool, default=False, metavar='FSRC',
                    help='flag to vary the number of input frames during training.')
parser.add_argument('--dataset_name', type=str, default='howto100m', metavar='DN',
                    help='specify training dataset')

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        args.video_dir = args.video_dir.replace('/research', '/net/ivcfs5/mnt/data')
        args.data_dir = args.data_dir.replace('/research', '/net/ivcfs5/mnt/data')
        
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 0
    args.multiprocessing_distributed = ngpus_per_node > 0
    args.dist_url = "tcp://localhost:23459"

    model_name = 'agg_%s_freeze_linear_%s_epochs_%s_bs_%s_warmup_steps_%s_init_lr_%s_accum_steps_%s_segments_%s_num_frames_%s_model' % (args.hierarchical_agg_function, args.freeze_llama_proj, args.num_epochs, args.batch_size, args.warmup_steps, args.init_lr, args.accum_grad_iters, args.num_segments, args.num_frames_per_clip)

    if 'linear' in args.hierarchical_agg_function:
        model_name = 'merged_weight_%s_' % args.global_region_embed_weight + model_name

    args.saved_checkpoints_dir = os.path.join(args.saved_checkpoints_dir, model_name)
    args.tensorboard_path = os.path.join(args.tensorboard_path, model_name)

    if not os.path.exists(args.saved_checkpoints_dir):
        os.mkdir(args.saved_checkpoints_dir)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)

    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
            print('args.rank: ', args.rank)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.num_gpus,
            rank=args.rank,
        )

    print('args.gpu: ', args.gpu)
    print('args.rank: ', args.rank)

    # instantiate the dataset and dataloader
    cfg = Config(args)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # instantiate the model, and load the pre-trained weights
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    if args.single_feature_per_frame:
        model.single_feature_per_frame = True

    if args.use_memory:
        model.use_memory = True
        model.num_memory_tokens = args.num_memory_tokens

    model.clip_level_generation = args.clip_level_generation
    model.num_frames_per_clip = args.num_frames_per_clip
    model.num_segments = args.num_segments
    model.hierarchical_agg_function = args.hierarchical_agg_function
    model.global_region_embed_weight = args.global_region_embed_weight

    model.initialize_visual_agg_function()

    if args.freeze_llama_proj:
        for k, v in model.llama_proj.named_parameters():
            v.requires_grad = False

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            #args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size = int(args.batch_size / args.num_gpus)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if args.dataset_name == 'howto100m':
        train_dataset = HowTo100MDataset(args, vis_processor, 'train', args.first_stage_random_clips)
    elif args.dataset_name == 'crosstask':
        train_dataset = CrossTaskDataset(args, vis_processor, 'train', args.first_stage_random_clips, model.module.llama_tokenizer)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    # initialize training hyperparameters and optimizer
    model_optimizer = get_optimizer(args, cfg, model)
    lr_scheduler = get_lr_scheduler(args, cfg, model_optimizer, len(train_dataset))

    # initialize gradient scaler in the case of mixed precision training
    amp = args.scaling_amp
    if amp:
        training_scaler = torch.cuda.amp.GradScaler()
    else:
        training_scalar = None

    # load pretrained checkpoint if path is specified
    if args.resume_training_checkpoint:
        checkpoint = torch.load(args.resume_training_checkpoint, map_location='cpu')
        latest_epoch = checkpoint['epoch']
        latest_model_state_dict = checkpoint['model_state_dict']
        latest_optimizer_state_dict = checkpoint['optimizer_state_dict']
        latest_gradscaler_state_dict = checkpoint['gradscaler_state_dict']
        latest_scheduler_state_dict = checkpoint['scheduler_state_dict']

        args.start_epoch = latest_epoch + 1

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in latest_model_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

        model_optimizer.load_state_dict(latest_optimizer_state_dict)
        training_scaler.load_state_dict(latest_gradscaler_state_dict)
        lr_scheduler.load_state_dict(latest_scheduler_state_dict)
        print('done loading resume checkpoint')
    else:
        args.start_epoch = 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )

    if args.rank == 0:
        # plot losses for training
        progress_writer = SummaryWriter(args.tensorboard_path)
    else:
        progress_writer = None

    train(args, train_loader, model, model_optimizer, lr_scheduler, training_scaler, progress_writer)

def train(args, train_loader, model, model_optimizer, lr_scheduler, training_scaler, progress_writer):
    model.train()
    if dist.get_rank() == 0:
        print('start training')
    for curr_epoch in range(args.start_epoch, args.num_epochs):
        curr_loss = train_single_epoch(args, curr_epoch, train_loader, model, model_optimizer, lr_scheduler, training_scaler, progress_writer, accum_grad_iters=args.accum_grad_iters)
        curr_checkpoint_path = os.path.join(args.saved_checkpoints_dir, 'epoch_%s_full.pth' % curr_epoch)

        if args.rank == 0:
            save_checkpoint(curr_epoch, model, model_optimizer, training_scaler, curr_checkpoint_path, lr_scheduler)

def train_single_epoch(args, epoch, train_loader, model, optimizer, lr_scheduler, scaler, progress_writer, accum_grad_iters=1):
    use_amp = scaler is not None
    
    ngpus_per_node = torch.cuda.device_count()
    num_steps = math.ceil(float(len(train_loader.dataset)) / (args.batch_size * ngpus_per_node))
    start_step = num_steps * epoch
    running_loss = 0.0
    s = time.time()

    if args.first_stage_random_clips:
        train_loader.dataset.random_sample_num_frames()

    for idx, data in enumerate(train_loader):
        if args.first_stage_random_clips:
            train_loader.dataset.random_sample_num_frames()
        
        #lr_scheduler.step(cur_epoch=epoch, cur_step=idx)
        lr_scheduler.step()

        s_step = time.time()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(data)['loss']

        d_step = time.time() - s_step
        running_loss += loss.item()

        # after_train_step()
        loss = loss / args.accum_grad_iters # add normalization to prevent gradient from becoming too large

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # update gradients every accum_grad_iters iterations
        if (idx + 1) % args.accum_grad_iters == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update() 
            else:
                optimizer.step()
            optimizer.zero_grad()

        if args.rank == 0:
            curr_step = start_step + idx
            progress_writer.add_scalar('Loss/train', loss, curr_step)

        if args.rank == 0 and idx % args.display_iter == 0 and idx != 0:
            d = time.time() - s
            curr_info = "Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f" % (
                                epoch,
                                d,
                                args.batch_size * dist.get_world_size() * float(idx) / len(train_loader.dataset),
                                running_loss / args.display_iter,
                                optimizer.param_groups[0]['lr'],
                            )
            running_loss = 0.0
            s = time.time()

            print(curr_info)


    return loss.item()

def get_lr_scheduler(args, cfg, optimizer, num_train_samples):
    max_epoch = args.num_epochs
    min_lr = args.min_lr
    init_lr = args.init_lr

    # optional parameters
    decay_rate = cfg.run_cfg.get("lr_decay_rate", None)
    warmup_start_lr = args.warmup_lr
    warmup_steps = args.warmup_steps

    iters_per_epoch = num_train_samples // int(args.batch_size*dist.get_world_size())
    final_num_train_samples = iters_per_epoch * int(args.batch_size*dist.get_world_size())
    final_num_training_steps = iters_per_epoch * max_epoch

    #lr_scheduler = LinearWarmupCosineLRScheduler(optimizer, max_epoch, iters_per_epoch, min_lr, init_lr, warmup_steps, warmup_start_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, final_num_training_steps)
    return lr_scheduler

def get_optimizer(args, cfg, model):
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        #print(n)
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()

    logging.info("number of trainable parameters: %d" % num_parameters)

    optim_params = [{"params": p_wd, "weight_decay": float(args.weight_decay),},
                    {"params": p_non_wd, "weight_decay": 0},]

    beta2 = cfg.run_cfg.get("beta2", 0.999)
    init_lr = args.init_lr
    model_optimizer = torch.optim.AdamW(
        optim_params,
        lr=init_lr,
        weight_decay=float(args.weight_decay),
        betas=(0.9, beta2),
    )

    return model_optimizer

def save_checkpoint(epoch, model, optimizer, scaler, path, lr_scheduler):
    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]

    torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'gradscaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            }, path)
    return

if __name__ == '__main__':
    main()
