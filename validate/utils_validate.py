import numpy as np
import torch
import torchvision.transforms as transforms
import os
import sys
import torchvision
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_config(args):
    if args.dataset_name == 'SC_imsize64':
        args.mean_norm, args.std_norm = [0.4706, 0.4601, 0.4549], [0.2750, 0.2754, 0.2837]
        args.ncls, args.input_size = 196, 64
        if args.model == 'ResNet18':
            args.adamw_lr = 1e-3
            args.eta = 1 if args.ipc == 10 else 2
        elif args.model == 'ResNet50':
            args.adamw_lr, args.eta = 1e-3, 1
        else:
            args.adamw_lr, args.eta = 5e-4, 2
    elif args.dataset_name == 'CUB_imsize64':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2150, 0.2107, 0.2483]
        args.ncls, args.input_size = 200, 64
        if args.model == 'ResNet18':
            args.adamw_lr = 1e-3
            args.eta = 1 if args.ipc == 10 else 2
        elif args.model == 'ResNet50':
            args.adamw_lr, args.eta = 1e-3, 1
        else:
            args.adamw_lr, args.eta = 5e-4, 2
    elif args.dataset_name == 'A_imsize64':
        args.mean_norm, args.std_norm = [0.4865, 0.5178, 0.5425], [0.2012, 0.1947, 0.2280]
        args.ncls, args.input_size = 100, 64
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 1e-3
            args.eta = 2 if args.ipc == 10 else 1
        else:
            args.adamw_lr, args.eta = 0.0005, 1
    elif args.dataset_name == 'SC_imsize224':
        args.mean_norm, args.std_norm = [0.4708, 0.4601, 0.4551], [0.2885, 0.2879, 0.2962]
        args.ncls, args.input_size = 196, 224
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 1e-3  # lr
            args.eta = 1 if args.ipc == 50 else 2
        elif args.model == 'ResNet101':
            # lr
            if args.ipc == 10 or args.ipc == 1:
                args.adamw_lr = 1e-3
            elif args.ipc == 50:
                args.adamw_lr = 5e-4
            args.eta = 2
        else:
            args.adamw_lr, args.eta = 5e-4, 1
    elif args.dataset_name == 'CUB_imsize224':
        args.mean_norm, args.std_norm = [0.4857, 0.4994, 0.4326], [0.2260, 0.2215, 0.2595]
        args.ncls, args.input_size = 200, 224
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 1e-3  # lr
            args.eta = 1 if args.ipc == 50 else 2
        elif args.model == 'ResNet101':
            # lr
            if args.ipc == 10 or args.ipc == 1:
                args.adamw_lr = 5e-3
            elif args.ipc == 50:
                args.adamw_lr = 5e-4
            args.eta = 2
        else:
            args.adamw_lr, args.eta = 5e-4, 1
    elif args.dataset_name == 'A_imsize224':
        args.mean_norm, args.std_norm = [0.4865, 0.5177, 0.5425], [0.2124, 0.2051, 0.2375]
        args.ncls, args.input_size = 100, 224
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 1e-3  # lr
            args.eta = 1 if args.ipc == 50 else 2
        elif args.model == 'ResNet101':
            # lr
            if args.ipc == 10 or args.ipc == 1:
                args.adamw_lr = 1e-3
            elif args.ipc == 50:
                args.adamw_lr = 5e-4
            args.eta = 2
        else:
            args.adamw_lr, args.eta = 5e-4, 1
    else:
        raise ValueError('dataset not supported')
    return args


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def result_append(epoch, epochs, visualization_metrics, train_loss, val_loss, train_lr, train_top1, val_top1,
                  train_top5, val_top5):
    epochs.append(epoch)
    train_loss.append(visualization_metrics["train/loss"])
    val_loss.append(visualization_metrics["val/loss"])
    train_lr.append(visualization_metrics["train/lr"])
    train_top1.append(visualization_metrics["train/Top1"])
    val_top1.append(visualization_metrics["val/top1"])
    train_top5.append(visualization_metrics["train/Top5"])
    val_top5.append(visualization_metrics["val/top5"])
    return epochs, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5, val_top5


def result_append_cal(epoch, epochs, visualization_metrics, train_lr,
                      train_loss_cal, val_loss_cal, train_loss_backbone, val_loss_backbone,
                      train_top1_cal, val_top1_cal, train_top1_backbone, val_top1_backbone,
                      train_top5_cal, val_top5_cal, train_top5_backbone, val_top5_backbone):
    epochs.append(epoch)
    # ==================================== loss ==============================================
    #                                      cal
    train_loss_cal.append(visualization_metrics["train/loss_cal"])
    val_loss_cal.append(visualization_metrics["val/loss_cal"])
    #                                    backbone
    train_loss_backbone.append(visualization_metrics["train/loss_backbone"])
    val_loss_backbone.append(visualization_metrics["val/loss_backbone"])
    # ===================================== lr ==============================================
    train_lr.append(visualization_metrics["train/lr"])
    # ==================================== Top1 ==============================================
    #                                      cal
    train_top1_cal.append(visualization_metrics["train/Top1_cal"])
    val_top1_cal.append(visualization_metrics["val/top1_cal"])
    #                                    backbone
    train_top1_backbone.append(visualization_metrics["train/Top1_backbone"])
    val_top1_backbone.append(visualization_metrics["val/top1_backbone"])
    # ==================================== Top5 ==============================================
    #                                      cal
    train_top5_cal.append(visualization_metrics["train/Top5_cal"])
    val_top5_cal.append(visualization_metrics["val/top5_cal"])
    #                                    backbone
    train_top5_backbone.append(visualization_metrics["train/Top5_backbone"])
    val_top5_backbone.append(visualization_metrics["val/top5_backbone"])
    return (epochs, train_lr,
            train_loss_cal, val_loss_cal, train_loss_backbone, val_loss_backbone,
            train_top1_cal, val_top1_cal, train_top1_backbone, val_top1_backbone,
            train_top5_cal, val_top5_cal, train_top5_backbone, val_top5_backbone)


def draw_result(epochs, train_loss, val_loss, train_lr, train_top1, val_top1, train_top5, val_top5, args):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    axes[0, 0].plot(epochs, train_loss, label="train/loss", marker=",", linestyle="-", color="blue",
                    linewidth=0.5)
    axes[0, 0].plot(epochs, val_loss, label="val/loss", marker=",", linestyle="-", color="green", linewidth=0.5)
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(False)

    axes[0, 1].plot(epochs, train_lr, label="train/lr", marker=",", linestyle="-", color="purple", linewidth=0.5)
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("train/lr")
    axes[0, 1].legend()
    axes[0, 1].grid(False)

    axes[1, 0].plot(epochs, train_top1, label="train/Top1", marker=",", linestyle="-", color="blue", linewidth=0.5)
    axes[1, 0].plot(epochs, val_top1, label="val/Top1", marker=",", linestyle="-", color="green", linewidth=0.5)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("Top1")
    axes[1, 0].legend()
    axes[1, 0].grid(False)

    axes[1, 1].plot(epochs, train_top5, label="train/Top5", marker=",", linestyle="-", color="blue", linewidth=0.5)
    axes[1, 1].plot(epochs, val_top5, label="val/Top5", marker=",", linestyle="-", color="green", linewidth=0.5)
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("Top5")
    axes[1, 1].legend()
    axes[1, 1].grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join("result", args.project, f"{args.exp_name}.png"))
    plt.close()


def draw_result_cal(args, epochs, train_lr,
                    train_loss_cal, val_loss_cal, train_loss_backbone, val_loss_backbone,
                    train_top1_cal, val_top1_cal, train_top1_backbone, val_top1_backbone,
                    train_top5_cal, val_top5_cal, train_top5_backbone, val_top5_backbone):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    # ========================================= loss ===========================================
    #                                        train loss
    axes[0, 0].plot(epochs, train_loss_cal, label=f"train-{args.model}_cal", marker=",", linestyle="-", color="blue",
                    linewidth=0.5)
    axes[0, 0].plot(epochs, train_loss_backbone, label=f"train-{args.model}", marker="o", linestyle="-", color="blue",
                    markersize=5, linewidth=0.5)
    train_loss = args.cal_ratio * np.array(train_loss_cal) + (1. - args.cal_ratio) * np.array(train_loss_backbone)
    axes[0, 0].plot(epochs, train_loss, label="train-total", marker="x", linestyle="-", color="blue",
                    markersize=7, linewidth=0.5)
    #                                          val loss
    axes[0, 0].plot(epochs, val_loss_cal, label=f"val-{args.model}_cal", marker=",", linestyle="-", color="green",
                    linewidth=0.5)
    axes[0, 0].plot(epochs, val_loss_backbone, label=f"val-{args.model}", marker="o", linestyle="-", color="green",
                    markersize=5, linewidth=0.5)
    val_loss = args.cal_ratio * np.array(val_loss_cal) + (1. - args.cal_ratio) * np.array(val_loss_backbone)
    axes[0, 0].plot(epochs, val_loss, label="val-total", marker="x", linestyle="-", color="green",
                    markersize=7, linewidth=0.5)
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(False)
    # ============================================ lr ===========================================
    axes[0, 1].plot(epochs, train_lr, label="lr", marker=",", linestyle="-", color="purple", linewidth=0.5)
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("lr")
    axes[0, 1].legend()
    axes[0, 1].grid(False)
    # ============================================ Top1 ===========================================
    #                                           train top1
    axes[1, 0].plot(epochs, train_top1_cal, label=f"train-{args.model}_cal", marker=",", linestyle="-", color="blue",
                    linewidth=0.5)
    axes[1, 0].plot(epochs, train_top1_backbone, label=f"train-{args.model}", marker="o", linestyle="-", color="blue",
                    markersize=5, linewidth=0.5)
    #                                            val top1
    axes[1, 0].plot(epochs, val_top1_cal, label=f"val-{args.model}_cal", marker=",", linestyle="-", color="green",
                    linewidth=0.5)
    axes[1, 0].plot(epochs, val_top1_backbone, label=f"val-{args.model}", marker="o", linestyle="-", color="green",
                    markersize=5, linewidth=0.5)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("Top1")
    axes[1, 0].legend()
    axes[1, 0].grid(False)
    # ============================================ Top5 ===========================================
    #                                           train top5
    axes[1, 1].plot(epochs, train_top5_cal, label=f"train-{args.model}_cal", marker=",", linestyle="-", color="blue",
                    linewidth=0.5)
    axes[1, 1].plot(epochs, train_top5_backbone, label=f"train-{args.model}", marker="o", linestyle="-", color="blue",
                    markersize=5, linewidth=0.5)
    #                                            val top5
    axes[1, 1].plot(epochs, val_top5_cal, label=f"val-{args.model}_cal", marker=",", linestyle="-", color="green",
                    linewidth=0.5)
    axes[1, 1].plot(epochs, val_top5_backbone, label=f"val-{args.model}", marker="o", linestyle="-", color="green",
                    markersize=5, linewidth=0.5)
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("Top5")
    axes[1, 1].legend()
    axes[1, 1].grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join("result", args.project, f"{args.exp_name}.png"))
    plt.close()


# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.cnt = None
        self.sum = None
        self.avg = None
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model, cal=None):
    group_no_weight_decay, group_weight_decay = [], []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)

    if cal is not None:
        for pname, p in cal.named_parameters():
            if pname.find('weight') >= 0 and len(p.size()) > 1:
                # print('include ', pname, p.size())
                group_weight_decay.append(p)
            else:
                # print('not include ', pname, p.size())
                group_no_weight_decay.append(p)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups


def load_small_dataset_model(model, args):
    net = None
    if model == 'ResNet18':
        net = ResNet18(args.ncls)
    elif model == 'ResNet50':
        net = ResNet50(args.ncls)
    elif model == 'ResNet101':
        net = ResNet101(args.ncls)
    return net


def load_val_loader(args):
    if args.dataset_name == "cifar100" or args.dataset_name == "cifar10" or args.dataset_name == "tiny_imagenet":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name == "imagenet1k" or args.dataset_name == "imagenet100" or args.dataset_name == 'imagenet-nette':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name in ("CUB_imsize64", "CUB_imsize224", "A_imsize64", "A_imsize224",
                               "SC_imsize64", "SC_imsize224"):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    else:
        raise NotImplementedError(f"dataset {args.dataset_name} not implemented")

    test_set = torchvision.datasets.ImageFolder(root=args.val_dir, transform=transform_test)

    # load dataset
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    return testloader
