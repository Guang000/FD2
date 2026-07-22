import argparse
import math
import os
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils_validate import AverageMeter, accuracy, get_parameters, load_val_loader, result_append, draw_result
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from models.utils_models import load_model, batch_augment
from relabel.utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX, RandomHorizontalFlipWithRes,
                               RandomResizedCropWithCoords, mix_aug)


def get_args():
    parser = argparse.ArgumentParser("FKD Training")
    parser.add_argument('--exp_name', type=str, default="", help='the name of the run')
    parser.add_argument('--original_data_path', required=True, type=str, help='name of the original data')
    parser.add_argument('--simple', default=False, action='store_true')
    parser.add_argument('--fkd_path', required=True, type=str, help='path to the fkd labels')
    parser.add_argument('--fkd_source', required=True, default="backbone", choices=["backbone", "cal"],
                        type=str, help="select the fkd labels's source")
    parser.add_argument('--save', default=False, action='store_true', help='save output')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('--dataset_name', default='cifar100', type=str, help='dataset name')
    parser.add_argument('--min_scale', type=float, default=0.08)
    parser.add_argument('--batch_size', type=int, default=16, help='CUB/A_imsize64:20, SC_imsize64:14')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--ipc', type=int, help='number of images per class')
    parser.add_argument('--cos', default=False, action='store_true', help='cosine lr scheduler')
    parser.add_argument('--eta', type=float, default=2.0, help='cosine lr scheduler eta')
    parser.add_argument('--multistep', default=False, action='store_true', help='multistep lr scheduler')
    parser.add_argument('--rate', type=float, default=0.9, help='multistep lr scheduler rate')
    parser.add_argument('--duration', type=float, default=2.0, help='multistep lr scheduler duration')
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # SGD optimizer else AdamW
    parser.add_argument('--sgd', default=False, action='store_true', help='sgd optimizer')
    parser.add_argument('--momentum', type=float, default=0.5, help='sgd momentum')  # checked
    parser.add_argument('--model', type=str, default='ResNet18', help='student model name')
    parser.add_argument('--model_source', type=str, default='CVDD', choices=["CVDD", "torchvision"])
    parser.add_argument('--keep_topk', type=int, default=1000, help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float, default=3.0, help='temperature for distillation loss')
    # Visualization
    parser.add_argument('--project', type=str, default='RankDD', help='project name')
    parser.add_argument('--matplotlib', action="store_true", help='whether to use matplotlib')
    parser.add_argument('--mix_type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int, help='seed for batch loading sampler')
    parser.add_argument('--val_dir', required=True, type=str, help="path to the validation data")
    args = parser.parse_args()
    args.mode = 'fkd_load'
    # final checked
    if args.dataset_name == 'SC_imsize64':
        args.mean_norm, args.std_norm = [0.4706, 0.4601, 0.4549], [0.2750, 0.2754, 0.2837]
        args.ncls, args.input_size = 196, 64
    elif args.dataset_name == 'CUB_imsize64':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2150, 0.2107, 0.2483]
        args.ncls, args.input_size = 200, 64
    elif args.dataset_name == 'A_imsize64':
        args.mean_norm, args.std_norm = [0.4865, 0.5178, 0.5425], [0.2012, 0.1947, 0.2280]
        args.ncls, args.input_size = 100, 64
    elif args.dataset_name == 'SC_imsize224':
        args.mean_norm, args.std_norm = [0.4708, 0.4601, 0.4551], [0.2885, 0.2879, 0.2962]
        args.ncls, args.input_size = 196, 224
    elif args.dataset_name == 'CUB_imsize224':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2260, 0.2215, 0.2595]
        args.ncls, args.input_size = 200, 224
    elif args.dataset_name == 'A_imsize224':
        args.mean_norm, args.std_norm = [0.4865, 0.5177, 0.5425], [0.2124, 0.2051, 0.2375]
        args.ncls, args.input_size = 100, 224
    elif args.dataset_name == 'imagenette_imsize224':
        args.mean_norm, args.std_norm = [0.4625, 0.4580, 0.4297], [0.2846, 0.2809, 0.3036]
        args.ncls, args.input_size = 10, 224
    elif args.dataset_name == 'imagewoof_imsize224':
        args.mean_norm, args.std_norm = [0.4855, 0.4559, 0.3934], [0.2591, 0.2513, 0.2602]
        args.ncls, args.input_size = 10, 224
    else:
        raise ValueError('dataset not supported')

    # set up the train_dir and output_dir.
    args.output_dir = os.path.join(str(args.output_dir), args.dataset_name, args.exp_name)
    print(f"Validate process args:\n{args}")
    return args


def is_special_epoch(epoch, total_epochs):
    in_last_80_percent = epoch >= int(total_epochs * 0.8) 
    ends_with_9_or_last = (epoch % 10 == 9) or (epoch == total_epochs - 1)  
    return in_last_80_percent and ends_with_9_or_last


def main():
    args = get_args()

    if args.matplotlib:
        os.makedirs(os.path.join("result", args.project), exist_ok=True)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    print(f"args.original_data_path: {args.original_data_path}")
    assert os.path.exists(args.original_data_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data loading
    train_dataset = ImageFolder_FKD_MIX(fkd_path=args.fkd_path, mode=args.mode, args_epoch=args.epochs,
                                        args_bs=args.batch_size, root=args.original_data_path,
                                        transform=ComposeWithCoords(transforms=[
                                            RandomResizedCropWithCoords(size=args.input_size, scale=(args.min_scale, 1),
                                                                        interpolation=InterpolationMode.BILINEAR),
                                            RandomHorizontalFlipWithRes(), transforms.ToTensor(),
                                            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
                                        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)

    args.train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=args.workers,
        pin_memory=False)

    # load validation data
    args.val_loader = load_val_loader(args)

    if args.model_source not in ["CVDD", "torchvision"]:
        raise ValueError(f"Now model_source only support CVDD or torchvision, your model_source is {args.model_source}")
    # load student model
    model = load_model(args.model, args.ncls, args.model_source, False, False).cuda()
    print(f"=> Load student model {args.model} and cal from source {args.model_source} successfully!")
    model.train()
    if args.sgd:
        args.optimizer = torch.optim.SGD(get_parameters(model, cal=None), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
    else:
        args.optimizer = torch.optim.AdamW(get_parameters(model, cal=None), lr=args.lr,
                                           weight_decay=args.weight_decay)

    if args.cos:
        args.scheduler = LambdaLR(args.optimizer,
                                  lambda step: 0.5 * (1. + math.cos(
                                      math.pi * step / args.epochs / args.eta)) if step <= args.epochs else 0,
                                  last_epoch=-1)
    elif args.multistep:
        args.scheduler = LambdaLR(args.optimizer, lambda step: args.rate ** (step / args.duration), last_epoch=-1)
    else:
        args.scheduler = LambdaLR(args.optimizer,
                                  lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0, last_epoch=-1)

    # args.best_acc1, args.optimizer, args.scheduler = 0, optimizer, scheduler
    args.best_acc1 = 0
    epochs, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5, val_top5 = [], [], [], [], [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):

        global visualization_metrics
        visualization_metrics = {}

        train(model, args, epoch)
        if args.simple:
            top1 = validate(model, args, epoch) if is_special_epoch(epoch, args.epochs) else 0
        else:
            top1 = validate(model, args, epoch) if (epoch % 10 == 0 or epoch == args.epochs - 1) else 0

        if args.matplotlib and (epoch % 10 == 0 or epoch == args.epochs - 1):
            epochs, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5, val_top5 = result_append(
                epoch, epochs, visualization_metrics, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5,
                val_top5)
            draw_result(epochs, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5, val_top5, args)
        args.scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        if args.save:
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc1': args.best_acc1,
                             'optimizer': args.optimizer.state_dict(), 'scheduler': args.scheduler.state_dict()},
                            is_best, output_dir=args.output_dir)

def train(model, args, epoch=None):
    objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    # optimizer = args.optimizer
    # scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean').cuda()
    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    # print(f"\nEpoch: {epoch}")
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]
        soft_label_cal, soft_label_backbone = soft_label
        images, target = images.cuda(), target.cuda()
        if args.fkd_source == "backbone":
            soft_label = soft_label_backbone.cuda().float()
        elif args.fkd_source == "cal":
            soft_label = soft_label_cal.cuda().float()
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        args.optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] usually isn't equal to args.batch_size in the last batch
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]
            # =================================== forward ===================================
            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))
            # ============================== processing soft labels ==============================
            output = F.log_softmax(output / args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label / args.temperature, dim=1)
            # ================================== loss ==================================
            loss = loss_function_kl(output, partial_soft_label)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        args.optimizer.step()

    metrics = {"train/loss": objs.avg, "train/Top1": top1.avg,
               "train/Top5": top5.avg,
               "train/lr": args.scheduler.get_last_lr()[0], "train/epoch": epoch, }
    if args.matplotlib:
        visualization_metrics.update(metrics)

    # print(f'Train\n'
    #       f"{args.fkd_source}'s fkd:loss={objs.avg:.6f},Top1={top1.avg:.2f},Top5={top5.avg:.2f}\n"
    #       f'train_time={time.time() - t1:.2f},lr={args.scheduler.get_last_lr()[0]:.6f}')


def validate(model, args, epoch=None):
    objs, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    t1 = time.time()
    with (torch.no_grad()):
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    print(f"Test epoch {epoch}:  {args.fkd_source}'s fkd:loss={objs.avg:.6f},Top1={top1.avg:.2f},Top5={top5.avg:.2f},val_time={time.time() - t1:.2f}")

    metrics = {'val/loss': objs.avg, 'val/top1': top1.avg,
               'val/top5': top5.avg, 'val/epoch': epoch, }
    if args.matplotlib:
        visualization_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    path = os.path.join(output_dir, 'ckpt.pth.tar' if epoch is None else f'ckpt_epoch{epoch}.pth.tar')
    torch.save(state, path)

    if is_best:
        path_best = os.path.join(output_dir, f'ckpt_epoch{epoch}_best_model.pth.tar')
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method('spawn')
    main()
