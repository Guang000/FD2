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
    # Load the patches
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
        # print(f"class{i}: {len(all_image_files)} images")
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
    initialised_data = np.transpose(initialised_data, (0, 3, 1, 2))  # Now shape is (N, C, H, W)
    N, C, _, _ = initialised_data.shape
    init_input_size = args.input_size_lis[0]
    # Downsample if needed
    if init_input_size != args.input_size:
        downsampled_data = np.zeros((N, C, init_input_size, init_input_size), dtype=np.float32)
        for i in range(N):
            for j in range(C):
                downsampled_data[i, j] = cv2.resize(initialised_data[i, j], (init_input_size, init_input_size),
                                                    interpolation=cv2.INTER_LINEAR)
        print("Downsampled the data")
    else:
        downsampled_data = initialised_data
    # convert the data to tensor
    patch_data = torch.tensor(downsampled_data, dtype=torch.float, requires_grad=True, device=device)
    init_data = torch.tensor(initialised_data, dtype=torch.float, device=device)
    return patch_data, init_data 


def load_recover_model(recover_model_name_list, args, device):
    all_recover_model_list, BN_hooks, model_sources = [], [], []
    for curr_recover_model_name in recover_model_name_list:
        curr_recover_model, model_source = None, None
        if args.pretrained_model_type == 'offline':
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
        # online model loading
        else: 
            curr_recover_model = load_online_model(curr_recover_model_name, args)
            model_source = "torchvision"

        model_sources.append(model_source)
        curr_recover_model = curr_recover_model.to(device)
        # freeze the compare model
        curr_recover_model.eval() 
        for p in curr_recover_model.parameters():  # 冻结权重
            p.requires_grad = False
        all_recover_model_list.append(curr_recover_model)

        # Process BN features
        curr_BN_hook = []
        for module in curr_recover_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                curr_BN_hook.append(BNFeatureHook(module))  
        BN_hooks.append(curr_BN_hook)  

    return all_recover_model_list, BN_hooks, model_sources

def to_sci_text(x):
    s = f"{x:.0e}"        # 0.5 -> '5e-01', 1.0 -> '1e+00'
    base, exp = s.split("e")
    exp = int(exp)
    if exp <= 0:
        return f"{base}e-{abs(exp)}"
    else:
        return f"{base}e{exp}"

def load_recover_model_cal(recover_model_name_list, args, device):
    all_recover_model_list, BN_hooks, model_sources, all_recover_cal_list, feature_centers = [], [], [], [], []
    for i, curr_recover_model_name in enumerate(recover_model_name_list):
        curr_recover_model, model_source, state_dict = None, None, None
        if args.pretrained_model_type == 'offline':
            curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + f'_M{args.M[i]}_{to_sci_text(args.cal_ratio[i])}cal.pth')
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
        cal = CAL(args.ncls, args.M[i], curr_recover_model_name, model_source)
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

    return all_recover_model_list, BN_hooks, model_sources, all_recover_cal_list, feature_centers


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


def phase_loss_append(phase_index, phase_iteration, curr_iter, phase_CE_loss, loss_ce, phase_BN_loss, loss_BN,
                      phase_total_loss, total_loss):
    phase_iteration[f'{phase_index}'].append(curr_iter)
    phase_CE_loss[f'{phase_index}'].append(loss_ce)
    phase_BN_loss[f'{phase_index}'].append(loss_BN)
    phase_total_loss[f'{phase_index}'].append(total_loss)
    return phase_iteration, phase_CE_loss, phase_BN_loss, phase_total_loss


# 2 Fine Grained Loss
def phase_loss_append_cal(num_call, args, phase_index, phase_iteration, curr_iter, phase_CE_loss_cal, loss_ce_cal,
                          phase_CE_loss_backbone, loss_ce_backbone, phase_BN_loss, curr_loss_BN, phase_total_loss, loss,
                          phase_FC_loss=None, fc_loss=None, phase_SC_loss=None, SC_loss_now_model=None,
                          phase_SC_loss_thresholds=None):
    phase_iteration[f'{phase_index}'].append(curr_iter)
    phase_CE_loss_cal[f'{phase_index}'].append(loss_ce_cal)
    phase_CE_loss_backbone[f'{phase_index}'].append(loss_ce_backbone)
    phase_BN_loss[f'{phase_index}'].append(curr_loss_BN)
    phase_total_loss[f'{phase_index}'].append(loss)
    if args.FC:
        phase_FC_loss[f'{phase_index}'].append(fc_loss)
    if args.SC and num_call != args.ipc_start:
        phase_SC_loss[f'{phase_index}'].append(SC_loss_now_model)
        phase_SC_loss_thresholds[f'{phase_index}'].append(args.SC_loss_threshold)

    return (phase_iteration, phase_CE_loss_cal, phase_CE_loss_backbone, phase_BN_loss, phase_total_loss,
            phase_FC_loss, phase_SC_loss, phase_SC_loss_thresholds)

def draw_loss(start_label, num_call, args, phase_CE_loss, phase_BN_loss, phase_total_loss,
              phase_iteration, phase_index):
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=False)
    for pha in range(phase_index + 1):
        axes[0].plot(phase_iteration[f"{pha}"], phase_CE_loss[f"{pha}"], linewidth=0.5, label="CE_loss", marker=",",
                     linestyle="-", color=colors[pha])
        axes[1].plot(phase_iteration[f"{pha}"], phase_BN_loss[f"{pha}"], linewidth=0.5, label="BN_loss", marker=",",
                     linestyle="-", color=colors[pha])
        axes[2].plot(phase_iteration[f"{pha}"], phase_total_loss[f"{pha}"], linewidth=0.5, label="total_loss",
                     marker=",",
                     linestyle="-", color=colors[pha])

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("CE_loss")
    axes[0].legend()
    axes[0].grid(False)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("BN_loss")
    axes[1].legend()
    axes[1].grid(False)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("total_loss")
    axes[2].legend()
    axes[2].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("result", f"{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                             f"cls{start_label}_{min(start_label + args.class_num, args.ncls) - 1}",
                             "recover_base_loss.png"))
    plt.close()


# 2 Fine Grained Loss
def draw_loss_cal(start_label, num_call, args, phase_CE_loss_cal, phase_CE_loss_backbone,
                  phase_BN_loss, phase_total_loss, phase_FC_loss, phase_SC_loss, phase_SC_loss_thresholds,
                  phase_iteration, phase_index):
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=False)
    for pha in range(phase_index + 1):
        axes[1, 0].plot(phase_iteration[f"{pha}"], phase_CE_loss_cal[f"{pha}"], linewidth=0.5, label="CE loss cal",
                        marker=",", linestyle="-", color=colors[pha])
        axes[1, 0].plot(phase_iteration[f"{pha}"], phase_CE_loss_backbone[f"{pha}"], linewidth=0.5,
                        label="CE loss backbone", marker="*", linestyle="-", color=colors[pha])
        axes[1, 1].plot(phase_iteration[f"{pha}"], phase_BN_loss[f"{pha}"], linewidth=0.5, label="BN loss", marker=",",
                        linestyle="-", color=colors[pha])
        axes[1, 2].plot(phase_iteration[f"{pha}"], phase_total_loss[f"{pha}"], linewidth=0.5, label="total loss cal",
                        marker=",", linestyle="-", color=colors[pha])
        if args.FC:
            axes[0, 0].plot(phase_iteration[f"{pha}"], phase_FC_loss[f"{pha}"], linewidth=0.5, label="FC loss cal",
                            marker=",", linestyle="-", color=colors[pha])
        if args.SC and num_call != args.ipc_start:
            axes[0, 1].plot(phase_iteration[f"{pha}"], phase_SC_loss[f"{pha}"], linewidth=0.5,
                            label="SC loss cal", marker=",", linestyle="-", color=colors[pha])
            axes[0, 1].plot(phase_iteration[f"{pha}"], phase_SC_loss_thresholds[f"{pha}"], linewidth=0.5,
                            label="SC loss thresholds", marker=",", linestyle="-", color="black")

    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("CE loss")
    axes[1, 0].legend()
    axes[1, 0].grid(False)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("BN loss")
    axes[1, 1].legend()
    axes[1, 1].grid(False)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("total loss")
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
        norms = F.normalize(feat_flat, p=2, dim=2, eps=eps)  
        sim = torch.bmm(norms, norms.transpose(1, 2)) 
        dist = (1.0 - sim) / 2.0 
    elif method.lower() == "l1":
        dist_nume = torch.cdist(feat_flat, feat_flat, p=1)  
        norms = feat_flat.norm(p=1, dim=2)  
        dist_deno = norms.unsqueeze(1) + norms.unsqueeze(2) + eps 
        dist = dist_nume / dist_deno 
    elif method.lower() == "l2":
        dist_nume = torch.cdist(feat_flat, feat_flat, p=2)  
        norms = feat_flat.norm(p=2, dim=2)
        dist_deno = norms.unsqueeze(1) + norms.unsqueeze(2) + eps  
        dist = dist_nume / dist_deno 

    idx_i, idx_j = torch.triu_indices(c, c, offset=1, device=features.device)
    dist_pairs = dist[:, idx_i, idx_j] 
    mean_per_image = dist_pairs.mean(dim=1)
    avg_dist = mean_per_image.mean()

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
    if method.lower() not in ["cos", "l1", "l2"]:
        raise ValueError(f"Now type only support cos, l1, l2. Your type is {method}")
    if target.lower() not in ["up", "down"]:
        raise ValueError(f"Now target only support up, down. Your target is {target}")
    if targets.shape[-1] != source.shape[-1] or targets.shape[-2] != source.shape[-2]:
        targets = F.interpolate(targets, size=source.shape[-2:], mode='bilinear', align_corners=False)
    b, c, h, w = targets.shape

    tar = targets.view(b, c * h * w)
    src = source.view(1, c * h * w).expand(b, c * h * w)
    dist = None
    if method.lower() == "cos":
        sim = F.cosine_similarity(tar, src, dim=1, eps=eps) 
        dist = (1.0 - sim) / 2.0 
    elif method.lower() == "l1":
        dist_nume = torch.norm(tar - src, p=1, dim=1)  
        norm_tar = torch.norm(tar, p=1, dim=1)  # [b]
        norm_src = torch.norm(src, p=1, dim=1)  # [b]
        dist_deno = norm_tar + norm_src + eps  # [b]
        dist = dist_nume / dist_deno  # [b]
    elif method.lower() == "l2":
        dist_nume = torch.norm(tar - src, p=2, dim=1)  
        norm_tar = torch.norm(tar, p=2, dim=1)  # [b]
        norm_src = torch.norm(src, p=2, dim=1)  # [b]
        dist_deno = norm_tar + norm_src + eps  # [b]
        dist = dist_nume / dist_deno  # [b]

    avg_dist = dist.mean()
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
    """
    Args:
        feature_matrix: Tensor of shape [B, M*C]
        labels:         LongTensor of shape [B], 
        feature_center: Tensor of shape [num_classes, M*C]
        method:           "l2", "l1" or "cos"
        target:         the target of process feature center distance

    Returns:
        distances: Tensor of shape [B]
    """
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
        lab = labels[i].item()  

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
