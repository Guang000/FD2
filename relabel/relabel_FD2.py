import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils_fkd import ComposeWithCoords, ImageFolder_FKD_MIX, RandomHorizontalFlipWithRes, RandomResizedCropWithCoords, mix_aug, load_model_cal, count_jpg_files
from models.utils_models import LastFeatureHook, get_module
import platform
import sys
import gc
from tqdm import tqdm

# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

parser = argparse.ArgumentParser(description='FKD Soft Label Generation w/ Mix Augmentation')
parser.add_argument('--syn_data_path', required=True, type=str,
                    help='the path to the syn data which is being processed in this relabeling process')
parser.add_argument('--model_choice', nargs='+', type=str, help='A list containing the choices of the compare model')
parser.add_argument('--model_weight', nargs='+', type=float, help='A list containing the choices of the compare model')
parser.add_argument('--M', nargs='+', type=int, help="cal's attention's number")
parser.add_argument('--cal_ratio', nargs='+', type=float, help="cal's ratio in CE loss")
parser.add_argument('--eval_mode', action='store_true', help='whether to use the evaluation mode or not')
parser.add_argument('--model_pool_dir', type=str, default=None,
                    help='required when pretrained model type is offline, '
                         'the directory of the models when using offline mode')
parser.add_argument('--fkd_path', required=True, type=str, help='the path to save the fkd soft labels')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel, CUB/A_imsize64:20, SC_imsize64:14')
parser.add_argument('--dataset_name', default='cifar100', type=str, help='dataset name')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
# FKD soft label generation args
parser.add_argument('--start_epochs', default=0, type=int)
parser.add_argument('--end_epochs', default=10000, type=int)
parser.add_argument("--min_scale_crops", type=float, default=0.08, help="argument in RandomResizedCrop")
parser.add_argument("--max_scale_crops", type=float, default=1., help="argument in RandomResizedCrop")
parser.add_argument('--use_fp16', dest='use_fp16', action='store_true', help='save soft labels as `fp16`')
parser.add_argument('--mode', default='fkd_save', type=str, metavar='N')
parser.add_argument('--fkd_seed', default=42, type=int, metavar='N')
parser.add_argument('--mix_type', default=None, type=str, choices=['mixup', 'cutmix', None],
                    help='mixup or cutmix or None')
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')


def set_worker_sharing_strategy(worker_id: int) -> None:
    if platform.system() == 'Linux':
        sharing_strategy = 'file_descriptor'
    else:
        sharing_strategy = 'file_system'
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    args = parser.parse_args()

    # set up the mean, std and ncls for the dataset
    if args.dataset_name == 'cifar100':
        args.mean_norm, args.std_norm = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        args.ncls, args.input_size = 100, 32
    elif args.dataset_name == 'cifar10':
        args.mean_norm, args.std_norm = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        args.ncls, args.input_size = 10, 32
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm, args.std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        args.ncls, args.input_size = 1000, 224
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm, args.std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        args.ncls, args.input_size = 10, 224
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm, args.std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        args.ncls, args.jitter, args.input_size = 200, 4, 64
    elif args.dataset_name == 'imagenet100':
        args.mean_norm, args.std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        args.ncls, args.jitter, args.input_size = 100, 32, 224
    elif args.dataset_name == 'NewStanfordCars10':
        args.mean_norm, args.std_norm = [0.4705, 0.4601, 0.4549], [0.2619, 0.2633, 0.2712]
        args.ncls, args.jitter, args.input_size = 10, 4, 32
    elif args.dataset_name == 'SC_imsize64':
        args.mean_norm, args.std_norm = [0.4706, 0.4601, 0.4549], [0.2750, 0.2754, 0.2837]
        args.ncls, args.jitter, args.input_size = 196, 4, 64
    elif args.dataset_name == 'CUB_imsize64':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2150, 0.2107, 0.2483]
        args.ncls, args.jitter, args.input_size = 200, 4, 64
    elif args.dataset_name == 'A_imsize64':
        args.mean_norm, args.std_norm = [0.4865, 0.5178, 0.5425], [0.2012, 0.1947, 0.2280]
        args.ncls, args.jitter, args.input_size = 100, 4, 64
    elif args.dataset_name == 'SC_imsize224':
        args.mean_norm, args.std_norm = [0.4708, 0.4601, 0.4551], [0.2885, 0.2879, 0.2962]
        args.ncls, args.jitter, args.input_size = 196, 32, 224
    elif args.dataset_name == 'CUB_imsize224':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2260, 0.2215, 0.2595]
        args.ncls, args.jitter, args.input_size = 200, 32, 224
    elif args.dataset_name == 'A_imsize224':
        args.mean_norm, args.std_norm = [0.4865, 0.5177, 0.5425], [0.2124, 0.2051, 0.2375]
        args.ncls, args.jitter, args.input_size = 100, 32, 224
    elif args.dataset_name == 'imagenette_imsize224':
        args.mean_norm, args.std_norm = [0.4625, 0.4580, 0.4297], [0.2846, 0.2809, 0.3036]
        args.ncls, args.input_size = 10, 224
    elif args.dataset_name == 'imagewoof_imsize224':
        args.mean_norm, args.std_norm = [0.4855, 0.4559, 0.3934], [0.2591, 0.2513, 0.2602]
        args.ncls, args.input_size = 10, 224
    else:
        raise ValueError('dataset not supported')
    # compute current ipc
    ipc = int(count_jpg_files(args.syn_data_path) / args.ncls)
    # set up the fkd path
    args.fkd_path = args.fkd_path + f'_bs{args.batch_size}_ipc{ipc}'
    os.makedirs(args.fkd_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting from checkpoints.')

    main_worker(args)


def main_worker(args):
    # load pretrained different teacher models
    teacher_model_lis, teacher_cal_lis, last_feature_hooks = [], [], []
    for model_id, model_name in enumerate(args.model_choice):
        model, cal, source = load_model_cal(args, model_id, model_name)
        model = model.cuda()
        teacher_model_lis.append(model)
        module = get_module(model_name, source, model)
        last_feature_hook = LastFeatureHook(module)
        last_feature_hooks.append(last_feature_hook)
        cal = cal.cuda()
        teacher_cal_lis.append(cal)

    # freeze all layers
    for _model in teacher_model_lis:
        for name, param in _model.named_parameters():
            param.requires_grad = False
    for _cal in teacher_cal_lis:
        for name, param in _cal.named_parameters():
            param.requires_grad = False

    cudnn.benchmark = True

    print("process data from {}".format(args.syn_data_path))

    # normalize = transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size, scale=(args.min_scale_crops, args.max_scale_crops),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm),
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
    for epoch in tqdm(range(args.start_epochs, args.end_epochs)):
        dir_path = os.path.join(args.fkd_path, 'epoch_{}'.format(epoch))
        os.makedirs(dir_path, exist_ok=True)

        with torch.no_grad():
            if args.model_weight is None:
                weights = [1.0 / len(teacher_model_lis)] * len(teacher_model_lis)
            else:
                w, temperature = np.array([float(w) for w in args.model_weight]), 10
                w = w / temperature
                weights = np.exp(w) / np.sum(np.exp(w))

            """Generate soft labels and save"""
            for batch_idx, (images, target, flip_status, coords_status) in enumerate(train_loader):
                images = images.cuda()
                split_point = int(images.shape[0] // 2)
                origin_images = images
                images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)
                total_p_cal, total_p_backbone = [], []
                for idx, (_model, _cal) in enumerate(zip(teacher_model_lis, teacher_cal_lis)):
                    if args.eval_mode:
                        _model.eval()
                        _cal.eval()
                    cat_p_cal, cat_p_backbone = [], []
                    p_backbone = _model(origin_images[:split_point])
                    cat_p_backbone.append(p_backbone)
                    last_feature = last_feature_hooks[idx].feature
                    p_raw_cal, p_eff, feature_matrix, attention_map, attention_maps = _cal(last_feature)
                    cat_p_cal.append(p_raw_cal)
                    p_backbone = _model(origin_images[split_point:])
                    cat_p_backbone.append(p_backbone)
                    last_feature = last_feature_hooks[idx].feature
                    p_raw_cal, p_eff, feature_matrix, attention_map, attention_maps = _cal(last_feature)
                    cat_p_cal.append(p_raw_cal)
                    p_backbone = torch.cat(cat_p_backbone, 0) * weights[idx]
                    p_cal = torch.cat(cat_p_cal, 0) * weights[idx]
                    total_p_backbone.append(p_backbone)
                    total_p_cal.append(p_cal)
                p_backbone = torch.stack(total_p_backbone, 0)
                p_backbone = p_backbone.sum(0)
                p_cal = torch.stack(total_p_cal, 0)
                p_cal = p_cal.sum(0)
                if args.use_fp16:
                    p_backbone, p_cal = p_backbone.half(), p_cal.half()
                p_backbone, p_cal = p_backbone.unsqueeze(0).cpu(), p_cal.unsqueeze(0).cpu()
                p = torch.cat([p_cal, p_backbone], dim=0)
                batch_config = [coords_status, flip_status, mix_index, mix_lam, mix_bbox, p.cpu()]
                batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
                torch.save(batch_config, batch_config_path)
                for last_feature_hook in last_feature_hooks:
                    last_feature_hook.close()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
