import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from utils_recover_FADRM import (inter_feature_center_dist, intra_feature_center_dist, features_dist,
                                 phase_loss_append_cal, draw_loss_cal)
import utils_recover_FADRM as utils_re
from models.utils_models import LastFeatureHook, get_module
import time
import gc


def recover(task_info):
    num_call, targets, device, transform, aug, args = task_info
    print(f"currently processing label from {targets[0].item()} to {targets[-1].item()}")
    targets = targets.to(device)

    # Visualization
    if args.matplotlib:
        result_path = os.path.join("result", f"cal_{args.exp_name}_{args.dataset_name}_ipcID{num_call}",
                                   f"cls{targets[0].item()}_{targets[-1].item()}")
        os.makedirs(result_path, exist_ok=True)

    recover_model_list, BN_hooks, model_sources, recover_cal_list, feature_centers \
        = utils_re.load_recover_model_cal(args.model_choice, args, device)
    print(f"---The recover models are {', '.join(model_name for model_name in args.model_choice)}")
    last_feature_hooks, syned_intra_attentions = {}, {}
    for i, (model_name, model_source, model) in enumerate(zip(args.model_choice, model_sources, recover_model_list)):
        module = get_module(model_name, model_source, model)
        last_feature_hooks[i] = LastFeatureHook(module)
        syned_intra_attentions[i] = []  # [args.class_num, B, M, AH, AW]

    if args.SC and num_call != args.ipc_start:
        for target in targets:
            syned_intra_imgs_path = os.path.join(f"{args.syn_data_path}", f"new{target:03d}")
            syned_intra_imgs_name = [f"class{target:03d}_id{ipc_idx:03d}.jpg" for ipc_idx in
                                     range(args.ipc_start, num_call)]
            syned_intra_imgs = []
            for syned_intra_img_name in syned_intra_imgs_name:
                syned_intra_img = read_image(os.path.join(syned_intra_imgs_path, syned_intra_img_name))
                syned_intra_img = transform(syned_intra_img)
                syned_intra_img = aug(syned_intra_img) if args.apply_data_augmentation else syned_intra_img
                syned_intra_imgs.append(syned_intra_img)
            syned_intra_imgs = torch.stack(syned_intra_imgs, dim=0).to(device)  
            for i, (model_name, model, cal) in enumerate(zip(args.model_choice, recover_model_list, recover_cal_list)):
                syned_intra_outputs = model(syned_intra_imgs)
                syned_intra_feature = last_feature_hooks[i].feature
                p_raw_syned, p_eff_syned, feature_matrix_syned, attention_map_syned, attention_maps_syned = cal(
                    syned_intra_feature)
                syned_intra_attentions[i].append(attention_maps_syned)

    # initialization
    if args.initialisation_method == "Guassian":
        syning_imgs = torch.randn((targets.shape[0], 3, args.input_size_lis[0], args.input_size_lis[0]),
                                  requires_grad=True, device=device, dtype=torch.float)
        orig_patch = torch.randn((targets.shape[0], 3, args.input_size, args.input_size),
                                 device=device, dtype=torch.float).to(device)
        print("initialisation method: Guassian")
    else:  
        syning_imgs, orig_patch = utils_re.initialize_patch_data(targets[0].item(), targets[-1].item() + 1, args,
                                                                 num_call, device)
        print(f"initialisation method: Patches--{args.patch_diff} ")
    scaler = torch.amp.GradScaler('cuda')  # Initialize GradScaler for mixed precision training
    iteration_all = sum(args.optimization_budgets)

    lr_scheduler = utils_re.lr_cosine_policy(args.lr, 0, iteration_all)
    criterion = nn.CrossEntropyLoss().to(device)
    start_time = time.time()

    phase_index, model_counter, curr_iter = 0, 0, 0  
    start_input_size = args.input_size_lis[phase_index]
    print(f"---The start input size is {start_input_size}")
    # matplotlib Visualization
    save_every, phase_CE_loss_cal, phase_CE_loss_backbone, phase_BN_loss, phase_total_loss = 50, {}, {}, {}, {}
    phase_SC_loss_thresholds, phase_FC_loss, phase_SC_loss, phase_iteration = {}, {}, {}, {}
    for phase in range(len(args.optimization_budgets)):
        phase_iteration[f'{phase}'], phase_SC_loss_thresholds[f'{phase}'] = [], []
        phase_CE_loss_cal[f'{phase}'], phase_CE_loss_backbone[f'{phase}'], phase_BN_loss[f'{phase}'] = [], [], []
        phase_FC_loss[f'{phase}'], phase_SC_loss[f'{phase}'], phase_total_loss[f'{phase}'] = [], [], []
    
    id_number_distilled_time, delta_time = 0, 0
    for iteration_per_layer in args.optimization_budgets:
        number_distilled_time = 0
        optimizer = optim.Adam([{'params': [syning_imgs], 'lr': args.lr}], betas=(0.5, 0.9), eps=1e-8)
        for iteration in range(iteration_per_layer):
            start_time_optimize_distill_samples = time.time()
            
            model = recover_model_list[model_counter]
            cal = recover_cal_list[model_counter]
            model_name = args.model_choice[model_counter]
            BN_hook = BN_hooks[model_counter]
            feature_center = feature_centers[model_counter]

            lr_scheduler(optimizer, curr_iter, curr_iter)
            aug = transforms.Compose(
                [transforms.RandomResizedCrop(start_input_size), transforms.RandomHorizontalFlip()])
            syning_imgs_jit = aug(syning_imgs) if args.apply_data_augmentation else syning_imgs  

            off1, off2 = random.randint(0, args.jitter), random.randint(0, args.jitter)
            syning_imgs_jit = torch.roll(syning_imgs_jit, shifts=(off1, off2), dims=(2, 3))  

            optimizer.zero_grad()
            with (((torch.amp.autocast(device_type='cuda')))):
                outputs_backbone = model(syning_imgs_jit)
                loss_ce_backbone = criterion(outputs_backbone, targets)
                last_feature = last_feature_hooks[model_counter].feature
                p_raw, p_eff, feature_matrix, attention_map, attention_maps = cal(last_feature)
                loss_ce_cal = criterion(p_raw, targets)
                rescale_backbone = [args.first_bn_multiplier] + [1. for _ in range(len(BN_hook["backbone"]) - 1)]
                curr_loss_BN_backbone = sum(
                    [mod.r_feature * rescale_backbone[idx] for idx, mod in enumerate(BN_hook["backbone"])])
                rescale_cal = [1. for _ in range(len(BN_hook["cal"]))]
                curr_loss_BN_cal = sum([mod.r_feature * rescale_cal[idx] for idx, mod in enumerate(BN_hook["cal"])])
                curr_loss_BN = curr_loss_BN_backbone if args.cal_ratio[model_counter] == 0.0 else curr_loss_BN_backbone + curr_loss_BN_cal
                loss = args.cal_ratio[model_counter] * loss_ce_cal + \
                       (1. - args.cal_ratio[model_counter]) * loss_ce_backbone + \
                       args.r_bn * curr_loss_BN
                fc_loss = None
                if args.FC:
                    intra_FC_loss = intra_feature_center_dist(feature_matrix, targets, feature_center, 'l2', 'down')
                    inter_FC_loss = inter_feature_center_dist(feature_matrix, targets, feature_center, 'l2', 'up')
                    fc_loss = args.IntraFC_ratio * intra_FC_loss + (1.0 - args.IntraFC_ratio) * inter_FC_loss
                    loss += args.FC_ratio * fc_loss
                SC_loss_now_model = torch.tensor(0.0, device=device)
                if args.SC and num_call != args.ipc_start:
                    for i, target in enumerate(targets):
                        SC_loss_now_cls = features_dist(syned_intra_attentions[model_counter][i],
                                                              attention_maps[i].unsqueeze(dim=0),
                                                              method="l2", target="up", eps=1e-6)
                        SC_loss_now_model += SC_loss_now_cls
                    SC_loss_now_model /= len(targets)
                    if SC_loss_now_model.item() < args.SC_loss_threshold:
                        pass
                    else:
                        loss += args.SC_ratio * SC_loss_now_model
            model_counter = (model_counter + 1) % len(recover_model_list)
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            scaler.step(optimizer)  # Update model parameters
            if number_distilled_time < 5:
                delta_time = delta_time + (time.time() - start_time_optimize_distill_samples)
                number_distilled_time, id_number_distilled_time = number_distilled_time + 1, id_number_distilled_time + 1

            scaler.update()  # Adjust scaling facto
            syning_imgs.data = utils_re.clip(syning_imgs.data, args) 
            # Visualization
            if curr_iter % save_every == 0: 
                end_time = time.time()
                print(
                    f"-------ipcID:{num_call},phase:{phase_index},class:{targets[0].item()}_{targets[-1].item()},"
                    f"iteration:{curr_iter}-------")
                print(
                    f"Model:{model_name},total loss:{loss.item():.4f},ce loss:backbone={loss_ce_backbone.item():.4f},"
                    f"cal={loss_ce_cal.item():.4f},bn loss:{curr_loss_BN.item():.4f}")
                if args.FC: print(f"FC_loss: {fc_loss.item():.4f}")
                if args.SC and num_call != args.ipc_start:
                    print(f"SC_loss: {SC_loss_now_model.item():.4f}")

                if args.matplotlib:
                    (phase_iteration, phase_CE_loss_cal, phase_CE_loss_backbone, phase_BN_loss, phase_total_loss,
                     phase_FC_loss, phase_SC_loss, phase_SC_loss_thresholds) \
                        = phase_loss_append_cal(num_call, args, phase_index, phase_iteration, curr_iter,
                                                phase_CE_loss_cal, loss_ce_cal.item(), phase_CE_loss_backbone,
                                                loss_ce_backbone.item(), phase_BN_loss, curr_loss_BN.item(),
                                                phase_total_loss, loss.item(), phase_FC_loss, fc_loss.item(),
                                                phase_SC_loss, SC_loss_now_model.item(),
                                                phase_SC_loss_thresholds)
                    draw_loss_cal(targets[0].item(), num_call, args, phase_CE_loss_cal, phase_CE_loss_backbone,
                                  phase_BN_loss, phase_total_loss, phase_FC_loss, phase_SC_loss,
                                  phase_SC_loss_thresholds, phase_iteration, phase_index)

                print(f'time for previous iterations: {end_time - start_time:.4f}')
                start_time = time.time()
            curr_iter += 1 
        phase_index += 1
        syning_imgs = syning_imgs.detach()
        if curr_iter == iteration_all:
            continue
        else:
            curr_size = args.input_size_lis[phase_index]
            if curr_size != start_input_size:
                print("I changed")
                syning_imgs = F.interpolate(syning_imgs, size=(curr_size, curr_size), mode='bilinear',
                                            align_corners=False)
            if curr_size != orig_patch.shape[2]:
                print("I changed")
                patch_resize = F.interpolate(orig_patch, size=(curr_size, curr_size), mode='bilinear',
                                             align_corners=False)
            else:
                patch_resize = orig_patch
            syning_imgs = args.alpha * syning_imgs + (1 - args.alpha) * patch_resize
            print("Residual added")
        start_input_size = curr_size
        syning_imgs.requires_grad = True
        
    print(f"Average time for distill {args.class_num} images ({args.input_size}) of ipcID({num_call}) = {delta_time / id_number_distilled_time} s")
    if args.store_best_images:
        best_syning_imgs = utils_re.denormalize(syning_imgs.data.clone(), args)
        save_images(args, best_syning_imgs, targets, num_call)

    for i in range(len(args.model_choice)):
        last_feature_hooks[i].close()
    for BN_hook in BN_hooks:
        for mod in BN_hook["backbone"]: mod.close()
        for mod in BN_hook["cal"]: mod.close()
    del recover_model_list
    torch.cuda.empty_cache()  


def get_images_parallel(args, device, num_call, is_first_ipc):
    torch.cuda.empty_cache()
    print("get_images_parallel call")
    transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(mean=args.mean_norm, std=args.std_norm)])
    aug = transforms.Compose([transforms.RandomResizedCrop(args.input_size), transforms.RandomHorizontalFlip()])
    targets_all = torch.arange(args.ncls, dtype=torch.long) 

    start_index = args.start_index if is_first_ipc else 0  

    tasks_info = []
    for start_label in range(start_index, args.ncls, args.class_num):
        targets = targets_all[start_label:min(start_label + args.class_num, args.ncls)]
        tasks_info.append([num_call, targets, device, transform, aug, args])

    print(f"recover_base start: {args.subprocess_num} processes, {len(tasks_info)} tasks")
    with torch.multiprocessing.Pool(processes=args.subprocess_num) as pool:
        pool.map(recover, tasks_info)
    gc.collect()
    torch.cuda.empty_cache()


def save_images(args, images, targets, ipc_id):
    print("save_images call")
    for id in range(images.shape[0]):
        class_id = targets[id].item() if targets.ndimension() == 1 else targets[id].argmax().item()
        os.makedirs(args.syn_data_path, exist_ok=True)

        # save into separate folders
        dir_path = os.path.join(args.syn_data_path, f"new{class_id:03d}")
        os.makedirs(dir_path, exist_ok=True)
        # place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id)
        place_to_store = os.path.join(dir_path, f"class{class_id:03d}_id{ipc_id:03d}.jpg")

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def parse_args():
    parser = argparse.ArgumentParser("Recover data from pre-trained model using FADRM")
    # Visualization
    parser.add_argument('--matplotlib', action='store_true', help='whether to use matplotlib')
    # Overall Configs
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset to recover, currently support CIFAR-10, CIFAR-100, Tiny-ImageNet,'
                             'ImageNet-Nette, ImageNet-1k')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--apply_data_augmentation', action='store_true',
                        help='whether or not to apply data augmentation')
    parser.add_argument('--start_index', type=int, default=0,
                        help='start index of the class to recover')
    parser.add_argument('--optimization_budgets', nargs='+', type=int)
    parser.add_argument('--input_size_lis', nargs='+', type=int, default=[200, 224, 200, 224],
                        help='list of teacher models to recover')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--pretrained_model_type', type=str, required=True, choices=['offline', 'online'],
                        help='Offline: the models are pre-trained and stored in the model pool directory\
                              Online: the pretrained models are loaded by downloading from the Pytorch Official Models')
    parser.add_argument('--model_choice', nargs='+', help='The trained backbone model list, '
                                                          'ResNet18, ResNet50, DenseNet121, ShuffleNetV2, MobileNetV2')
    parser.add_argument('--M', nargs='+', type=int, help="cal's attention number")
    parser.add_argument('--cal_ratio', nargs='+', type=float, help="cal's ratio in CE loss")
    # Directory Related Configs
    parser.add_argument('--syn_data_path', type=str, required=True,
                        help='where to store synthetic data')
    parser.add_argument('--model_pool_dir', type=str, default=None,
                        help='required when pretrained model type is offline')
    parser.add_argument('--patch_dir', type=str, default=None,
                        help='the directory where the patches are stored')
    parser.add_argument('--initialisation_dir', default=None, type=str,
                        help="the directory of the initialisation data specifically for patch initialisation,\
                              it will create a sub folder named exp-name under this directory")
    # Data Saving Related Configs
    parser.add_argument('--store_best_images', action='store_true',
                        help='whether to store synthetic data')
    parser.add_argument('--store_initialised_images', action='store_true',
                        help='whether to store the initialised images when using patches initialisation')
    # Optimization Related Configs
    parser.add_argument('--class_num', type=int, default=1,
                        help='number of images to optimize in 1 process')
    parser.add_argument('--subprocess_num', type=int, default=10, help='number of recover_base subprocess')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r_bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--weight_temperature', default=5, type=int,
                        help="The temperature used when calculating the weight")
    # Fine-Grained Related Configs
    parser.add_argument('--FC', action='store_true', help='whether to use image semantic consistency loss')
    parser.add_argument('--FC_ratio', type=float, default=0.8, help='the ratio of fc_loss')
    parser.add_argument('--IntraFC_ratio', type=float, default=0.5, help='the ratio of intra_fc_loss')
    parser.add_argument('--SC', action='store_true',
                        help='whether to use intra class semantic diversity loss. '
                             'If use it, you must keep ipc_end - ipc_start > 1')
    parser.add_argument('--SC_ratio', type=float, default=0.2, help='the ratio of SC_loss')
    parser.add_argument('--SC_loss_threshold', type=float, default=0.6,
                        help='The smaller this parameter is, the greater the degree of change of the image will be,'
                             'range:[0.0, 1.0]')
    # Initialisation Related Configs
    parser.add_argument('--initialisation_method', type=str, default="Guassian", choices=["Guassian", "Patches"],
                        help='initialisation method for the synthetic data')
    parser.add_argument('--patch_diff', type=str, default="medium", choices=["1", "2"],
                        help="the difficulty of the patches")
    # IPC (Image Per Class) Related Configs
    parser.add_argument("--ipc_start", default=0, type=int, help="start index of IPC")
    parser.add_argument("--ipc_end", default=50, type=int, help="end index of IPC")
    args = parser.parse_args()

    # set up the path for the synthetic data
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    os.makedirs(args.syn_data_path, exist_ok=True)
    if args.dataset_name == 'SC_imsize64':
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
    else:
        raise ValueError('dataset not supported')

    return args


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    print("args\n", args)
    # set up device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"---The recover process will be performed on device: {device}")
    # loop through the IPCs and generate the synthetic data
    for ipc_id in range(args.ipc_start, args.ipc_end):
        start_time_ipc_id = time.time()
        get_images_parallel(args, device, ipc_id, is_first_ipc=(ipc_id == args.ipc_start))
        print(f"time for ipc_id({ipc_id}) = {time.time()-start_time_ipc_id}")
