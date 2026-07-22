import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import distributed, Tensor
import glob
import random
import os
import sys
import torchvision.models as models
import torch.optim as optim

# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from models.utils_models import load_model
import matplotlib

matplotlib.use('Agg')  # headless service
import matplotlib.pyplot as plt
import cv2


def get_second_idx(all_idx, exclude_idx):
    remaining_idx = [i for i in all_idx if i != exclude_idx]
    sample_idx = random.choice(remaining_idx)
    return sample_idx


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, args, use_fp16=False):
    """
    adjust the input based on mean and variance
    """
    if use_fp16:
        mean = np.array(args.mean_norm, dtype=np.float16)
        std = np.array(args.std_norm, dtype=np.float16)
    else:
        mean = np.array(args.mean_norm)
        std = np.array(args.std_norm)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, args, use_fp16=False):
    if use_fp16:
        mean = np.array(args.mean_norm, dtype=np.float16)
        std = np.array(args.std_norm, dtype=np.float16)
    else:
        mean = np.array(args.mean_norm)
        std = np.array(args.std_norm)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class BNFeatureHook: 
    def __init__(self, module):
        self.r_feature = None
        self.var = None
        self.mean = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        self.mean = mean
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        self.var = var
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


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


def load_verifier_model(chosen_name, args):
    model_path = os.path.join(args.model_pool_dir, chosen_name + ".pth")
    state_dict = torch.load(str(model_path), weights_only=True)
    model = None
    try:
        model = load_model(chosen_name, args.ncls, "CVDD", pretrained_weights=False)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        print(f"CVDD's {chosen_name} can't match ckpt, next try torchvision")
        try:
            model = load_model(chosen_name, args.ncls, "torchvision", False)
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            print(f"torchvision's {chosen_name} also can't match ckpt")
        else:
            print(f"torchvision's {chosen_name} can match ckpt")
    else:
        print(f"CVDD's {chosen_name} can match ckpt")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def normalize(image, args):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    normalized_image = (image - args.mean_norm) / args.std_norm
    return normalized_image


def initialize_patch_data(start_label_idx, end_label_idx, args, num_call, device="cuda"):
    initialisation_dir = None
    if args.store_initialised_images:
        initialisation_dir = os.path.join(args.initialisation_dir, args.exp_name, f'call_{num_call}',
                                          f'{start_label_idx}_to_{end_label_idx}')
        os.makedirs(initialisation_dir, exist_ok=True)
        print(f"Initialisation dir: {initialisation_dir}")
    # Load pre-made patches
    patch_dir = os.path.join(args.patch_dir, args.patch_diff) 

    all_images = []
    for i in range(start_label_idx, end_label_idx):
        current_class_dir = os.path.join(str(patch_dir), f"{i:05d}")
        if os.path.exists(current_class_dir):
            pass
            # print(f"class{i}_dir: {current_class_dir}")
        else:
            current_class_name = str(i)
            current_class_dir = os.path.join(str(patch_dir), current_class_name)
            print(f"class{i}_dir {current_class_dir} exist?: {os.path.exists(current_class_dir)}")
        all_image_files = glob.glob(f"{current_class_dir}/*.jpg", recursive=False)
        chosen_image_files = random.sample(all_image_files, 1)  

        final_img = normalize(cv2.imread(chosen_image_files[0]), args)
        final_img_display = cv2.imread(chosen_image_files[0])

        # save the img to the initialisation dir to show the quality of the patches
        # you can comment this line if you don't want to see the quality
        if args.store_initialised_images:
            new_img_file = os.path.join(str(initialisation_dir), f'{str(i)}.jpg')
            cv2.imwrite(new_img_file, final_img_display)
        # append the final image to the list
        all_images.append(final_img)

    # change the list to a numpy array
    initialised_data = np.array(all_images)
    initialised_data = np.transpose(initialised_data, (0, 3, 1, 2))
    # convert the data to tensor
    patch_data = torch.tensor(initialised_data, dtype=torch.float, requires_grad=True, device=device)  

    return patch_data


def load_recover_model(recover_model_name_list, args, device):
    all_recover_model_list, BN_hooks, weight_list, model_sources = [], [], [], []
    for curr_recover_model_name in recover_model_name_list:
        curr_recover_model, model_source = None, None
        if args.pretrained_model_type == 'offline':
            if args.dataset_name == 'imagenet100' or args.dataset_name == 'imagenet-nette':
                # code for imagenet100
                if curr_recover_model_name == 'ResNet18':
                    curr_recover_model = models.resnet18(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                elif curr_recover_model_name == 'ResNet50':
                    curr_recover_model = models.resnet50(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                elif curr_recover_model_name == 'Densenet121':
                    curr_recover_model = models.densenet121(weights=None)
                    in_features = curr_recover_model.classifier.in_features
                    curr_recover_model.classifier = torch.nn.Linear(in_features, args.ncls)
                elif curr_recover_model_name == 'MobileNetV2':
                    curr_recover_model = models.mobilenet_v2(weights=None)
                    in_features = curr_recover_model.classifier[-1].in_features
                    curr_recover_model.classifier[-1] = torch.nn.Linear(in_features, args.ncls)
                elif curr_recover_model_name == 'ShuffleNetV2':
                    curr_recover_model = models.shufflenet_v2_x1_0(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                else:
                    raise ValueError('Model not supported')
                model_source = "torchvision"
                curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + '.pth')
                state_dict = torch.load(str(curr_recover_model_weight_path), weights_only=False)
                curr_recover_model.load_state_dict(state_dict)
            # load process for cifar100, cifar10, and tinyimagenet
            else:
                curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + '.pth')
                state_dict = torch.load(str(curr_recover_model_weight_path), weights_only=True)
                try:
                    curr_recover_model = load_model(curr_recover_model_name, args.ncls, "CVDD", False)
                    curr_recover_model.load_state_dict(state_dict)
                except RuntimeError:
                    print(f"CVDD's {curr_recover_model_name} can't match ckpt, next try torchvision")
                    try:
                        curr_recover_model = load_model(curr_recover_model_name, args.ncls, "torchvision", False)
                        curr_recover_model.load_state_dict(state_dict)
                    except BaseException:
                        print(f"torchvision's {curr_recover_model_name} also can't match ckpt")
                    else:
                        model_source = "torchvision"
                        print(f"torchvision's {curr_recover_model_name} can match ckpt")
                else:
                    model_source = "CVDD"
                    print(f"CVDD's {curr_recover_model_name} can match ckpt")
        # online model loading for imagenet1k
        else: 
            curr_recover_model = load_online_model(curr_recover_model_name, args)
            model_source = "torchvision"

        model_sources.append(model_source)
        curr_recover_model = curr_recover_model.to(device)
        # freeze the compare model
        curr_recover_model.eval() 
        for p in curr_recover_model.parameters():  
            p.requires_grad = False
        all_recover_model_list.append(curr_recover_model)

        # Process BN features
        curr_BN_hook = []
        for module in curr_recover_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                curr_BN_hook.append(BNFeatureHook(module))
        BN_hooks.append(curr_BN_hook)

    if args.voter_type == 'equal':
        weight_list = [1 / len(recover_model_name_list)] * len(recover_model_name_list)  # ep. [0.5, 0.5]
    elif args.voter_type == 'prior':
        for model_name in recover_model_name_list:
            weight_list.append(args.model_prior_weight_dict[model_name])
        weight_list = np.array([float(w) for w in weight_list]) / args.weight_temperature
        weight_list = np.exp(weight_list) / np.sum(np.exp(weight_list))
    elif args.voter_type == 'random':
        random_list = np.random.rand(len(recover_model_name_list))
        normalized_list = random_list / random_list.sum()
        weight_list = normalized_list.tolist()
    else:
        raise ValueError(f"Voter type {args.voter_type} is not supported")

    return all_recover_model_list, BN_hooks, weight_list, model_sources

def to_sci_text(x):
    s = f"{x:.0e}"        # 0.5 -> '5e-01', 1.0 -> '1e+00'
    base, exp = s.split("e")
    exp = int(exp)
    if exp <= 0:
        return f"{base}e-{abs(exp)}"
    else:
        return f"{base}e{exp}"

def load_recover_model_cal(recover_model_name_list, M_list, args, device):
    all_recover_model_list, BN_hooks, weight_list, model_sources, all_recover_cal_list, feature_centers = [], [], [], [], [], []
    for i, curr_recover_model_name in enumerate(recover_model_name_list):
        curr_recover_model, model_source, state_dict = None, None, None
        if args.pretrained_model_type == 'offline':
            curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + f'_M{M_list[i]}_{to_sci_text(args.cal_ratio[i])}cal.pth')
            state_dict = torch.load(str(curr_recover_model_weight_path), weights_only=True)
            try:
                curr_recover_model = load_model(curr_recover_model_name, args.ncls, "CVDD", False)
                curr_recover_model.load_state_dict(state_dict[curr_recover_model_name])
            except RuntimeError:
                print(f"CVDD's {curr_recover_model_name} can't match ckpt, next try torchvision")
                try:
                    curr_recover_model = load_model(curr_recover_model_name, args.ncls, "torchvision", False)
                    curr_recover_model.load_state_dict(state_dict[curr_recover_model_name])
                except BaseException:
                    print(f"torchvision's {curr_recover_model_name} also can't match ckpt")
                else:
                    model_source = "torchvision"
                    print(f"torchvision's {curr_recover_model_name} can match ckpt")
            else:
                model_source = "CVDD"
                print(f"CVDD's {curr_recover_model_name} can match ckpt")
        # online model loading for imagenet1k
        else: 
            curr_recover_model = load_online_model(curr_recover_model_name, args)
            model_source = "torchvision"
        model_sources.append(model_source)
        curr_recover_model = curr_recover_model.to(device)
        cal = CAL(args.ncls, M_list[i], curr_recover_model_name, model_source)
        cal.load_state_dict(state_dict["cal"])
        cal = cal.to(device)
        # freeze the compare model
        curr_recover_model.eval()
        cal.eval()
        for p in curr_recover_model.parameters():
            p.requires_grad = False
        for p in cal.parameters():
            p.requires_grad = False
        all_recover_model_list.append(curr_recover_model)
        all_recover_cal_list.append(cal)

        # Process BN features
        curr_BN_hook = {"backbone": [], "cal": []}
        for module in curr_recover_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                curr_BN_hook["backbone"].append(BNFeatureHook(module))
        for module in cal.modules():
            if isinstance(module, nn.BatchNorm2d):
                curr_BN_hook["cal"].append(BNFeatureHook(module))
        BN_hooks.append(curr_BN_hook) 
        feature_center = state_dict["feature_center"].to(device)
        feature_centers.append(feature_center)

    if args.voter_type == 'equal':
        weight_list = [1 / len(recover_model_name_list)] * len(recover_model_name_list)  # ep. [0.5, 0.5]
    elif args.voter_type == 'prior':
        for model_name in recover_model_name_list:
            weight_list.append(args.model_prior_weight_dict[model_name])
        weight_list = np.array([float(w) for w in weight_list]) / args.weight_temperature
        weight_list = np.exp(weight_list) / np.sum(np.exp(weight_list)) 
    elif args.voter_type == 'random':
        random_list = np.random.rand(len(recover_model_name_list))
        normalized_list = random_list / random_list.sum()
        weight_list = normalized_list.tolist()
    else:
        raise ValueError(f"Voter type {args.voter_type} is not supported")

    return all_recover_model_list, BN_hooks, weight_list, model_sources, all_recover_cal_list, feature_centers


def load_online_model(model_name, args):
    if args.dataset_name == 'imagenet1k':
        if model_name == 'MobileNetV2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'Densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == 'EfficientNet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'ShuffleNetV2':
            model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} is not supported")
    else:
        raise NotImplementedError(f"Online model loading for {args.dataset_name} is not supported yet")

    return model


def draw_InterCSD_loss(recover_model_name_list, iteration_plt, total_InterCSD_loss_plt, model_InterCSD_loss_plt, args,
                       start_label, num_call):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=False)
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, recover_model_name in enumerate(recover_model_name_list):
        axes[0].plot(iteration_plt, model_InterCSD_loss_plt[recover_model_name], linewidth=0.5,
                     label=recover_model_name, marker=",", linestyle="-", color=colors[i])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("model InterCSD loss")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(iteration_plt, total_InterCSD_loss_plt, linewidth=0.5, label="total_InterCSD_loss", marker=",",
                 linestyle="-", color="brown")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("total InterCSD loss")
    axes[1].legend()
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("result", f"{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "recover_InterCSD_loss.png"))
    plt.close()


def draw_InterCSD_loss_cal(recover_model_name_list, iteration_plt, total_InterCSD_loss_plt, model_InterCSD_loss_plt,
                           args,
                           start_label, num_call):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=False)
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, recover_model_name in enumerate(recover_model_name_list):
        axes[0].plot(iteration_plt, model_InterCSD_loss_plt[recover_model_name], linewidth=0.5,
                     label=recover_model_name, marker=",", linestyle="-", color=colors[i])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("model InterCSD loss")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(iteration_plt, total_InterCSD_loss_plt, linewidth=0.5, label="total_InterCSD_loss", marker=",",
                 linestyle="-", color="brown")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("total InterCSD loss")
    axes[1].legend()
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("result", f"cal_{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "recover_InterCSD_loss.png"))
    plt.close()


def draw_total_loss(iteration_plt, total_loss_plt, args, start_label, num_call):
    plt.figure(figsize=(6, 4))
    plt.plot(iteration_plt, total_loss_plt, label='total_loss', marker=",", linestyle="-",
             color="brown", linewidth=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("total_loss")
    plt.title("total_loss vs. Iteration")
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join("result", f"{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "total_loss.png"))
    plt.close()


def model_loss_append(args, recover_model_name_list, model_CE_loss, ce_lis, model_BN_loss, loss_BN_lis,
                      model_total_loss, weight_list):
    for model_id, recover_model_name in enumerate(recover_model_name_list):
        model_CE_loss[recover_model_name].append(ce_lis[model_id].item())
        model_BN_loss[recover_model_name].append(loss_BN_lis[model_id].item())
        model_total_loss[recover_model_name].append(weight_list[model_id] * (
                ce_lis[model_id].item() + args.r_bn * loss_BN_lis[model_id].item()))

    return model_CE_loss, model_BN_loss, model_total_loss

# 2 Fine Grained Loss
def model_loss_append_cal(num_call, args, recover_model_name_list, cal_ratio_list, model_CE_loss_cal,
                          model_CE_loss_backbone, ce_lis_cal, ce_lis_backbone, model_BN_loss, loss_BN_lis,
                          model_total_loss, weight_list, model_FC_loss, fc_lis, model_SC_loss, SC_lis):
    for model_id, recover_model_name in enumerate(recover_model_name_list):
        model_CE_loss_cal[recover_model_name].append(ce_lis_cal[model_id].item())
        model_CE_loss_backbone[recover_model_name].append(ce_lis_backbone[model_id].item())
        model_BN_loss[recover_model_name].append(loss_BN_lis[model_id].item())
        model_total_loss[recover_model_name].append(weight_list[model_id] * (
                cal_ratio_list[model_id] * ce_lis_cal[model_id].item() +
                (1. - cal_ratio_list[model_id]) * ce_lis_backbone[model_id].item()
                + args.r_bn * loss_BN_lis[model_id].item()))
        if args.FC:
            model_FC_loss[recover_model_name].append(fc_lis[model_id].item())
            model_total_loss[recover_model_name][-1] += (weight_list[model_id] * args.FC_ratio *
                                                         fc_lis[model_id].item())
        if args.SC and num_call != args.ipc_start:
            model_SC_loss[recover_model_name].append(SC_lis[model_id].item())
            if SC_lis[model_id].item() < args.SC_loss_threshold:
                pass
            else:
                model_total_loss[recover_model_name][-1] += (weight_list[model_id] * args.SC_ratio *
                                                             SC_lis[model_id].item())

        return (model_CE_loss_cal, model_CE_loss_backbone, model_BN_loss, model_total_loss, model_FC_loss,
                model_SC_loss)


def draw_loss(start_label, num_call, args, recover_model_name_list, model_CE_loss, model_BN_loss, model_total_loss,
              total_loss_plt, iteration_plt):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex=False)
    axes[0, 1].plot(iteration_plt, total_loss_plt, linewidth=0.5, label="total_loss", marker=",", linestyle="-",
                    color="brown")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("total_loss")
    axes[0, 1].legend()
    axes[0, 1].grid(False)

    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i in range(len(recover_model_name_list)):
        axes[1, 0].plot(iteration_plt, model_CE_loss[recover_model_name_list[i]], linewidth=0.5,
                        label=recover_model_name_list[i], marker=",", linestyle="-", color=colors[i])
        axes[1, 1].plot(iteration_plt, model_BN_loss[recover_model_name_list[i]], linewidth=0.5,
                        label=recover_model_name_list[i], marker=",", linestyle="-", color=colors[i])
        axes[0, 0].plot(iteration_plt, model_total_loss[recover_model_name_list[i]], linewidth=0.5,
                        label=recover_model_name_list[i], marker=",", linestyle="-", color=colors[i])
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("CE loss")
    axes[1, 0].legend()
    axes[1, 0].grid(False)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("BN loss")
    axes[1, 1].legend()
    axes[1, 1].grid(False)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("model total loss")
    axes[0, 0].legend()
    axes[0, 0].grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join("result", f"{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "recover_base_loss.png"))
    plt.close()


# 2 Fine Grained Loss
def draw_loss_cal(start_label, num_call, args, recover_model_name_list, cal_ratio_list, model_CE_loss_cal,
                  model_CE_loss_backbone, model_BN_loss, model_total_loss, model_FC_loss, model_SC_loss,
                  SC_loss_thresholds, total_loss_plt, iteration_plt):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=False)
    axes[0, 2].plot(iteration_plt, total_loss_plt, linewidth=0.5, label="total loss", marker=",", linestyle="-",
                    color="brown")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("total loss")
    axes[0, 2].legend()
    axes[0, 2].grid(False)

    if args.SC and num_call != args.ipc_start:
        axes[0, 1].plot(iteration_plt, SC_loss_thresholds, linewidth=0.5,
                        label="SC_loss_threshold", marker=",", linestyle="-", color="black")
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, model_name in enumerate(recover_model_name_list):
        # CE Loss
        axes[1, 0].plot(iteration_plt, model_CE_loss_cal[model_name], linewidth=0.5,
                        label=f'{model_name}_cal', marker=",", linestyle="-", color=colors[i])
        axes[1, 0].plot(iteration_plt, model_CE_loss_backbone[model_name], linewidth=0.5,
                        label=model_name, marker="o", markersize=1, linestyle="-", color=colors[i])
        ce_loss = cal_ratio_list[i] * np.array(model_CE_loss_cal[model_name]) + \
                  (1. - cal_ratio_list[i]) * np.array(model_CE_loss_backbone[model_name])
        axes[1, 0].plot(iteration_plt, ce_loss, linewidth=0.5, label=f'{model_name}_total', marker="x",
                        markersize=1, linestyle="-", color=colors[i])
        # BN Loss
        axes[1, 1].plot(iteration_plt, model_BN_loss[model_name], linewidth=0.5,
                        label=f'{model_name}_cal', marker=",", linestyle="-", color=colors[i])
        # model total loss
        axes[1, 2].plot(iteration_plt, model_total_loss[model_name], linewidth=0.5,
                        label=model_name, marker=",", linestyle="-", color=colors[i])

        if args.FC:
            axes[0, 0].plot(iteration_plt, model_FC_loss[model_name], linewidth=0.5,
                            label=f'{model_name}_cal', marker=",", linestyle="-", color=colors[i])
        if args.SC and num_call != args.ipc_start:
            axes[0, 1].plot(iteration_plt, model_SC_loss[model_name], linewidth=0.5,
                            label=f'{model_name}_cal', marker=",", linestyle="-", color=colors[i])

    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("CE loss")
    axes[1, 0].legend()
    axes[1, 0].grid(False)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("BN loss")
    axes[1, 1].legend()
    axes[1, 1].grid(False)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("total model loss")
    axes[1, 2].legend()
    axes[1, 2].grid(False)

    if args.FC:
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel(f"FC loss(ratio={args.FC_ratio})")
        axes[0, 0].legend()
        axes[0, 0].grid(False)
    if args.SC and num_call != args.ipc_start:
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel(f"SC loss(ratio={args.SC_ratio})")
        axes[0, 1].legend()
        axes[0, 1].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("result", f"cal_{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "recover_base_loss.png"))
    plt.close()

def make_group_loader(full_dataset, class_list, batch_size=32, num_workers=2):

    ids = [i for i, (image_path, label) in enumerate(full_dataset.samples) if full_dataset.classes[label] in class_list]
    subset = Subset(full_dataset, ids)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def extract_bn_stats(model: torch.nn.Module):
    stats = {"running_mean": [], "running_var": []}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            stats[f"running_mean"].append(module.running_mean.clone().cpu())
            stats[f"running_var"].append(module.running_var.clone().cpu())
    return stats


def lower_channels(feat: torch.Tensor, target_c: int) -> torch.Tensor:
    """
    将 feat [b, c, h, w] 分为 target_c 组通道，每组取平均，输出 [b, target_c, h, w]。
    分组规则：
      - group_size = C // target_c
      - remainder = C % target_c
      - 前 target_c-1 组各包含 channels_per_group 通道
      - 最后一组包含 channels_per_group + remainder 通道
    Args:
        feat: 输入特征图，shape = [b, c, h, w]
        target_c: 目标通道数
    Returns:
        out_feat: 输出特征图，shape = [b, target_c, h, w]
    """
    b, c, h, w = feat.shape
    if target_c > c:
        raise ValueError(f"target_c({target_c}) must be smaller than c({c})")

    channels_per_group = c // target_c  # channels per group
    remainder = c % target_c  # 余数

    sizes = [channels_per_group] * target_c  # 构造每组的通道数列表
    sizes[-1] += remainder  # 最后一组加上余数

    groups = torch.split(feat, sizes, dim=1)  # 按 sizes 拆分,list of [b, sizes[i], h, w]
    groups_mean = [g.mean(dim=1, keepdim=True) for g in groups]  # 对每组在通道维做mean,使维度为 [b,1,h,w]
    out_feat = torch.cat(groups_mean, dim=1)  # 拼接回 [b, target_c, h, w]
    return out_feat


def channels_dist(features: torch.Tensor, method="cos", target="up", eps: float = 1e-6) -> torch.Tensor:
    """
    Get distance between channels in a feature map. Finally, get mean distance between features.
    Args:
        features: Your target features map.
        method: Now this function only support cosine similarity, l1 distance, l2 distance.
        target: If you want to up distance, you need to choose "up", else choose "down".
        eps: A value to avoid tiny distance.

    Returns: Final loss.
    """
    if method.lower() not in ["cos", "l1", "l2"]:
        raise ValueError(f"Now type only support cos, l1, l2. Your type is {method}")
    if target.lower() not in ["up", "down"]:
        raise ValueError(f"Now target only support up, down. Your target is {target}")

    b, c, h, w = features.shape
    feat_flat = features.view(b, c, h * w)
    dist = None
    if method.lower() == "cos":
        norms = F.normalize(feat_flat, p=2, dim=2, eps=eps)  # l2 归一化到单位向量 -> [b, c, h*w]
        sim = torch.bmm(norms, norms.transpose(1, 2))  # 余弦相似度, sim[b,i,j]=<v_i,v_j>->[b,c,c]
        dist = (1.0 - sim) / 2.0  # 归一化, [b, c, c], ∈[0,1]
    elif method.lower() == "l1":
        dist_nume = torch.cdist(feat_flat, feat_flat, p=1)  # 分子:l1距离矩阵 -> [b, c, c]
        norms = feat_flat.norm(p=1, dim=2)  # L1范数norms[b,i]=||v_i||_1 -> [b, c]
        dist_deno = norms.unsqueeze(1) + norms.unsqueeze(2) + eps  # 分母||v_i||_1 + ||v_j||_1 + eps -> [b, c, c]
        dist = dist_nume / dist_deno  # 归一化 -> [b, c, c], ∈[0,1]
    elif method.lower() == "l2":
        dist_nume = torch.cdist(feat_flat, feat_flat, p=2)  # 分子:l2距离矩阵 -> [b, c, c]
        norms = feat_flat.norm(p=2, dim=2)  # 计算范数norms[b,i]=||v_i||_2 -> [b, c]
        dist_deno = norms.unsqueeze(1) + norms.unsqueeze(2) + eps  # 分母||v_i||_2 + ||v_j||_2 + eps -> [b, c, c]
        dist = dist_nume / dist_deno  # 归一化 -> [b, c, c], ∈[0,1]

    # 只取上三角i<j的元素,构造两个长度=c*(c-1)/2的索引向量
    idx_i, idx_j = torch.triu_indices(c, c, offset=1, device=features.device)
    dist_pairs = dist[:, idx_i, idx_j]  # 收集每张图的唯一通道对们的距离 -> [b, c*(c-1)/2]
    mean_per_image = dist_pairs.mean(dim=1)  # 每张图的平均距离 -> [b]
    avg_dist = mean_per_image.mean()  # 批次归一化:求所有图片平均,作为最终“平均通道距离”

    loss = None
    if method.lower() == "cos":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    if method.lower() == "l1":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    if method.lower() == "l2":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    return loss


def features_dist(targets: torch.Tensor, source: torch.Tensor, method: str = "cos", target: str = "up",
                  eps: float = 1e-6) \
        -> torch.Tensor:
    """
    计算源特征图与每个目标特征图的归一化距离，并返回批次距离和总和。

    Args:
        targets: Tensor [b, c, h, w]
        source:  Tensor [1, c, h, w]
        method:  "l2", "l1" 或 "cos"
        target: If you want to up distance, you need to choose "up", else choose "down".
        eps:     防止除零的小常数

    Returns:
        dist_norm:  Tensor [b]，每张图的归一化距离
        total_dist: Scalar Tensor，总距离 = dist_norm.sum()
    """
    if method.lower() not in ["cos", "l1", "l2"]:
        raise ValueError(f"Now type only support cos, l1, l2. Your type is {method}")
    if target.lower() not in ["up", "down"]:
        raise ValueError(f"Now target only support up, down. Your target is {target}")

    b, c, h, w = targets.shape

    # 展平到 [b, c*h*w] 和 [b, c*h*w]
    tar = targets.view(b, c * h * w)
    src = source.view(1, c * h * w).expand(b, c * h * w)
    dist = None
    if method.lower() == "cos":
        sim = F.cosine_similarity(tar, src, dim=1, eps=eps)  # 余弦相似度 s ∈ [-1,1], [b]
        dist = (1.0 - sim) / 2.0  # 归一化到 [0,1]: d̄ = (1 − s) / 2, [b]
    elif method.lower() == "l1":
        dist_nume = torch.norm(tar - src, p=1, dim=1)  # # 原始 L1 距离, [b]
        # 对称归一化到[0,1]： d̄ = d / (||tar||₁ + ||src||₁ + eps)
        norm_tar = torch.norm(tar, p=1, dim=1)  # [b]
        norm_src = torch.norm(src, p=1, dim=1)  # [b]
        dist_deno = norm_tar + norm_src + eps  # [b]
        dist = dist_nume / dist_deno  # [b]
    elif method.lower() == "l2":
        dist_nume = torch.norm(tar - src, p=2, dim=1)  # # 原始 L2 距离, [b]
        # 对称归一化到[0,1]： d̄ = d / (||tar||_2 + ||src||_2 + eps)
        norm_tar = torch.norm(tar, p=2, dim=1)  # [b]
        norm_src = torch.norm(src, p=2, dim=1)  # [b]
        dist_deno = norm_tar + norm_src + eps  # [b]
        dist = dist_nume / dist_deno  # [b]

    avg_dist = dist.mean()  # 标量
    loss = None
    if method.lower() == "cos":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    if method.lower() == "l1":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    if method.lower() == "l2":
        if target.lower() == "up":
            loss = 1.0 - avg_dist
        elif target.lower() == "down":
            loss = avg_dist
    return loss


def intra_feature_center_dist(feature_matrix: torch.Tensor, labels, feature_center: torch.Tensor, method: str = "cos",
                              target: str = "down", eps: float = 1e-6) -> torch.Tensor:
    centers = feature_center[labels]  

    if method.lower() == "l2":
        d = torch.norm(feature_matrix - centers, p=2, dim=1)  
        nf = torch.norm(feature_matrix, p=2, dim=1)  # [B]
        nc = torch.norm(centers, p=2, dim=1)  # [B]
        distances = d / (nf + nc + eps)
    elif method.lower() == "l1":
        d = torch.norm(feature_matrix - centers, p=1, dim=1) 
        nf = torch.norm(feature_matrix, p=1, dim=1)  # [B]
        nc = torch.norm(centers, p=1, dim=1)  # [B]
        distances = d / (nf + nc + eps)
    elif method.lower() == "cos":
        s = F.cosine_similarity(feature_matrix, centers, dim=1, eps=eps) 
        distances = (1.0 - s) / 2.0 
    else:
        raise ValueError("method must be one of 'l2','l1','cos'")

    if target == "up":
        loss = 1.0 - distances.mean()
    elif target == "down":
        loss = distances.mean()
    else:
        raise ValueError("target must be one of 'up','down'")

    return loss


def inter_feature_center_dist(feature_matrix: torch.Tensor, labels, feature_center: torch.Tensor, method: str = "cos",
                              target: str = "up", eps: float = 1e-6) -> torch.Tensor:

    B, C = feature_matrix.shape
    num_classes, _ = feature_center.shape
    distances = []

    for i in range(B):
        x = feature_matrix[i]  # [C]
        lab = labels[i].item()
        if lab == 0:
            centers = feature_center[1:]
        elif lab == num_classes - 1:
            centers = feature_center[:num_classes - 1]
        else:
            centers = torch.cat([feature_center[:lab], feature_center[lab + 1:]], dim=0)  # [num_classes-1, C]

        if method.lower() == "l2":
            num = torch.norm(x - centers, p=2, dim=1)  # [num_classes-1]
            denom = torch.norm(x, p=2) + centers.norm(p=2, dim=1) + eps
            d = num / denom  # [num_classes-1]
        elif method.lower() == "l1":
            num = torch.norm(x - centers, p=1, dim=1)  # [num_classes-1]
            denom = torch.norm(x, p=1) + centers.norm(p=1, dim=1) + eps
            d = num / denom  # [num_classes-1]
        elif method.lower() == "cos":
            # [num_classes-1], ∈[-1,1]
            sim = F.cosine_similarity(x.unsqueeze(0).expand_as(centers), centers, dim=1, eps=eps)
            d = (1.0 - sim) / 2.0  # [num_classes-1], ∈[0,1]
        else:
            raise ValueError("method must be 'l2','l1' or 'cos'")
        distances.append(d.mean())
    distances_tensor = torch.stack(distances, dim=0)
    if target == "up":
        loss = 1.0 - distances_tensor.mean()
    elif target == "down":
        loss = distances_tensor.mean()
    else:
        raise ValueError("target must be one of 'up','down'")

    return loss


# [K, C], [B, C], [B]
def inter_feature_matrix_dist(syned_features: torch.Tensor, syning_features: torch.Tensor, labels: torch.LongTensor,
                              method: str = "l2", target: str = "up", eps: float = 1e-6) -> Tensor:
    B, C = syning_features.shape
    K, _ = syned_features.shape
    dists = []

    for i in range(B):
        x = syning_features[i]  # [C]
        lab = labels[i].item()  # 标量 0..K-1

        if lab == 0:
            syned_other_features = syned_features[1:]  # [K-1, C]
        elif lab == K - 1:
            syned_other_features = syned_features[:K - 1]  # [K-1, C]
        else:
            syned_other_features = torch.cat([syned_features[:lab], syned_features[lab + 1:]], dim=0)  # [K-1, C]

        if method.lower() == "l2":
            num = torch.norm(x - syned_other_features, p=2, dim=1)  # [K-1]
            denom = torch.norm(x, p=2) + syned_other_features.norm(p=2, dim=1) + eps
            d = num / denom  # [K-1]

        elif method.lower() == "l1":
            num = torch.norm(x - syned_other_features, p=1, dim=1)  # [K-1]
            denom = torch.norm(x, p=1) + syned_other_features.norm(p=1, dim=1) + eps
            d = num / denom  # [K-1]

        elif method.lower() == "cos":
            sim = F.cosine_similarity(x.unsqueeze(0).expand_as(syned_other_features),
                                      syned_other_features, dim=1, eps=eps)  # [K-1], ∈[-1,1]
            d = (1.0 - sim) / 2.0  # [K-1], ∈[0,1]
        else:
            raise ValueError("method must be 'l2','l1' or 'cos'")
        dists.append(d)
    dists = torch.stack(dists, dim=0)
    mean_per_sample = dists.mean(dim=1)

    if target == "up":
        loss = 1.0 - mean_per_sample.mean()
    elif target == "down":
        loss = mean_per_sample.mean()
    else:
        raise ValueError("target must be one of 'up','down'")
    return loss
