import torch.nn as nn
import torch
import glob
import os
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from models.utils_models import batch_augment
from torch.utils.data import DistributedSampler
from functools import partial
import time
import matplotlib

matplotlib.use('Agg')  # headless service
import matplotlib.pyplot as plt


def worker_init(worker_id, base_seed, rank):
    seed = base_seed + rank * 10 + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# load CIFAR-100 dataset
def load_dataset(rank, args, mode="train"):
    # Define the transformation of the dataset
    if mode == "train":
        transform_train = transforms.Compose([
            # transforms.RandomCrop(args.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(args.mean_norm, args.std_norm)
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(args.mean_norm, args.std_norm)])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(args.mean_norm, args.std_norm)])

    val_dir = os.path.join(args.dataset_dir, 'test')
    train_dir = os.path.join(args.dataset_dir, 'train')

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

    # Set case for multi-gpu training
    if args.use_multi_gpu:
        train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    else:
        train_sampler = None

    worker_init_fun = partial(worker_init, base_seed=args.base_seed, rank=rank)

    # load train dataset
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2,
        worker_init_fn=worker_init_fun, pin_memory=True, drop_last=False)

    # load test dataset
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


def lr_lambda(epoch):
    base_rate, base_duration = 0.9, 2.0
    return pow(base_rate, epoch / base_duration)



def get_all_models(args):
    pth_files = glob.glob(f"{args.save_dir}/*.pth", recursive=False)
    return pth_files


# Evaluate the model
def evaluate_loader(model, criterion, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    acc = correct / total
    loss = total_loss / len(dataloader)
    return acc, loss


def evaluate_loader_cal(net, cal, data_loader, device, last_feature_hook, loss_container_cal, raw_metric_cal,
                        drop_metric_cal, loss_container_backbone, raw_metric_backbone, criterion):
    # metrics initialization
    loss_container_cal.reset()
    raw_metric_cal.reset()
    drop_metric_cal.reset()
    loss_container_backbone.reset()
    raw_metric_backbone.reset()
    # begin validation
    start_time = time.time()
    net.eval()
    cal.eval()
    epoch_loss_cal, epoch_loss_backbone, epoch_acc_cal, epoch_acc_backbone, aux_acc_cal = 0., 0., [0., 0.], [0., 0.], [0., 0.]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            # ================================= Raw Image =================================
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            last_feature = last_feature_hook.feature
            pred_raw, pred_eff, feature_matrix, attention_map, attention_maps = cal(last_feature)
            # ================================= Crop Image =================================
            crop_images = batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            outputs_crop = net(crop_images)
            last_feature_crop = last_feature_hook.feature
            pred_raw_crop, pred_eff_crop, feature_matrix_crop, attention_map_crop, attention_maps_crop = cal(
                last_feature_crop)
            # ================================= Final prediction =================================
            pred_raw_aux = (pred_raw + pred_raw_crop) / 2.
            pred_eff_aux = (pred_eff + pred_eff_crop) / 2.
            # loss
            batch_loss_backbone = criterion(outputs, labels)
            epoch_loss_backbone = epoch_loss_backbone + loss_container_cal(batch_loss_backbone.item())
            batch_loss_cal = criterion(pred_raw_aux, labels)
            epoch_loss_cal = epoch_loss_cal + loss_container_cal(batch_loss_cal.item())
            # metrics: top-1,5 acc
            epoch_acc_now_cal = raw_metric_cal(pred_raw_aux, labels)
            aux_acc_now_cal = drop_metric_cal(pred_eff_aux, labels)
            epoch_acc_now_backbone = raw_metric_backbone(outputs, labels)
            for ids in range(len((1, 5))):
                epoch_acc_cal[ids] = epoch_acc_cal[ids] + epoch_acc_now_cal[ids]
                aux_acc_cal[ids] = aux_acc_cal[ids] + aux_acc_now_cal[ids]
                epoch_acc_backbone[ids] = epoch_acc_backbone[ids] + epoch_acc_now_backbone[ids]

    # end of validation
    epoch_loss_cal, epoch_loss_backbone = epoch_loss_cal / len(data_loader), epoch_loss_backbone / len(data_loader)
    for ids in range(len((1, 5))):
        epoch_acc_cal[ids] = epoch_acc_cal[ids] / len(data_loader)
        aux_acc_cal[ids] = aux_acc_cal[ids] / len(data_loader)
        epoch_acc_backbone[ids] = epoch_acc_backbone[ids] / len(data_loader)

    end_time = time.time()

    return epoch_loss_cal, epoch_loss_backbone, epoch_acc_cal, epoch_acc_backbone, aux_acc_cal, end_time - start_time


def draw(exp_name, epochs, train_loss_list_cal, train_acc_list_cal, train_acc_aux_list_cal, test_loss_list_cal,
         test_acc_list_cal, test_acc_aux_list_cal, train_loss_list_backbone, train_acc_list_backbone,
         test_loss_list_backbone, test_acc_list_backbone, lr_list, dataset_name, model_source, cal_ratio,
         model_name=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=False)
    # ================================================== loss ==================================================
    #                                                 train loss
    axes[0, 0].plot(epochs, train_loss_list_cal, linewidth=0.5, label=f"Train-{model_name}_cal", marker=",",
                    linestyle="-", color='blue')
    axes[0, 0].plot(epochs, train_loss_list_backbone, linewidth=0.5, label=f"Train-{model_name}", marker="o",
                    markersize=5, linestyle="-", color='blue')
    train_loss = cal_ratio * np.array(train_loss_list_cal) + (1. - cal_ratio) * np.array(train_loss_list_backbone)
    axes[0, 0].plot(epochs, train_loss, linewidth=0.5, label="Train-total", marker="x", markersize=7,
                    linestyle="-", color='blue')
    #                                                  test loss
    axes[0, 0].plot(epochs, test_loss_list_cal, linewidth=0.5, label=f"Test-{model_name}_cal", marker=",",
                    linestyle="-", color='red')
    axes[0, 0].plot(epochs, test_loss_list_backbone, linewidth=0.5, label=f"Test-{model_name}", marker="o",
                    markersize=5, linestyle="-", color='red')
    test_loss = cal_ratio * np.array(test_loss_list_cal) + (1. - cal_ratio) * np.array(test_loss_list_backbone)
    axes[0, 0].plot(epochs, test_loss, linewidth=0.5, label="Test-total", marker="x", markersize=7,
                    linestyle="-", color='red')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(False)
    # ================================================== lr ==================================================
    axes[1, 0].plot(epochs, lr_list, linewidth=0.5, label="lr", marker=",", linestyle="-", color='green')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("lr")
    axes[1, 0].legend()
    axes[1, 0].grid(False)
    # ================================================== AccRaw ==================================================
    # train
    axes[0, 1].plot(epochs, train_acc_list_cal, linewidth=0.5, label=f"Train-{model_name}_cal", marker=",",
                    linestyle="-", color='blue')
    axes[0, 1].plot(epochs, train_acc_list_backbone, linewidth=0.5, label=f"Train-{model_name}", marker="o",
                    markersize=5, linestyle="-", color='blue')
    # test
    axes[0, 1].plot(epochs, test_acc_list_cal, linewidth=0.5, label=f"Test-{model_name}_cal", marker=",",
                    linestyle="-", color='red')
    axes[0, 1].plot(epochs, test_acc_list_backbone, linewidth=0.5, label=f"Test-{model_name}", marker="o",
                    markersize=5, linestyle="-", color='red')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AccRaw")
    axes[0, 1].legend()
    axes[0, 1].grid(False)
    # ================================================== AccAux ==================================================
    axes[1, 1].plot(epochs, train_acc_aux_list_cal, linewidth=0.5, label=f"Train-{model_name}_cal", marker=",",
                    linestyle="-", color='blue')
    axes[1, 1].plot(epochs, test_acc_aux_list_cal, linewidth=0.5, label=f"Test-{model_name}_cal", marker=",",
                    linestyle="-", color='red')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AccAux")
    axes[1, 1].legend()
    axes[1, 1].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("results", dataset_name, model_source, f"{exp_name}.png"))
    plt.close()
