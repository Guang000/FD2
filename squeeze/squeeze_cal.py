import torch
import argparse
from utils_squeeze import load_dataset, evaluate_loader_cal, draw
from models.utils_models import load_model, get_module, LastFeatureHook, AverageMeter, TopKAccuracyMetric, CenterLoss, batch_augment
import torch.nn as nn
import sys
import os
# for multiprocessing system
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from timm.data import Mixup
from models.cal import CAL


# for initialisign the dist environment
def setup_distributed_environment(master_addr='localhost', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"MASTER_ADDR set to {master_addr}, MASTER_PORT set to {master_port}")


# Define the arguments of the program and parse them from the command line
def parse_args():
    parser = argparse.ArgumentParser("Squeezing the models")
    parser.add_argument('--model_list', nargs='+', help='The trained model list, '
                                                        'ResNet18, ResNet50, DenseNet121, ShuffleNetV2, MobileNetV2')
    parser.add_argument('--model_source', nargs="+", help="The trained model's source. CVDD or torchvision")
    parser.add_argument('--pretrained_weights', action='store_true', help='whether to use pretrained weights')
    parser.add_argument('--pretrained_bn', action='store_true', help='whether to use pretrained bn')
    parser.add_argument('--load_uncal_weights', action='store_true',
                        help='load uncal weights as initial backbone weights')
    parser.add_argument('--M', type=int, default=32, help='cal attention numbers')
    parser.add_argument('--cal_ratio', type=float, default=1.0, help="cal's ratio")
    parser.add_argument('--exp_name', type=str, default='ResNet18_cal', help='the name of the experiment')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataset_dir', type=str, required=True, help='directory where the dataset are stored')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the trained models')
    parser.add_argument('--batch_size', type=int, default=128, help='number of images to optimize at the same time')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset to use for training')
    parser.add_argument('--epoch', type=int, default=200, help='num of iterations to optimize the target model')
    parser.add_argument('--stop_epoch', type=int, default=50, help='value of epoch to stop training')
    parser.add_argument('--lr', nargs='+', type=float, help='learning rate for optimization')
    parser.add_argument('--cos_lr', action='store_true', help='whether to use cos schedule of lr')
    parser.add_argument('--rate', type=float, default=0.9, help='multistep rate')
    parser.add_argument('--duration', type=float, default=2.0, help='multistep duration')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi_gpu_learning')
    parser.add_argument('--world_size', type=int, default=-1, help='available gpu num, enable it with --use_multi_gpu')
    parser.add_argument('--base_seed', type=int, default=42, help='The base seed which used to set different '
                                                                  'seed for every dataset worker per process '
                                                                  'when use multi_gpu')
    parser.add_argument('--master_port', type=str, default='12355', help='DDP port')
    # Visualization
    parser.add_argument('--matplotlib', action='store_true', help="use matplotlib or not")
    args = parser.parse_args()

    if len(args.model_list) != len(args.model_source):
        raise ValueError('The num of model_source and model_list should be equal')

    # set up the mean, std and ncls for the dataset
    if args.dataset_name == 'cifar100':
        args.mean_norm, args.std_norm = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        args.ncls, args.input_size = 100, 32
    elif args.dataset_name == 'cifar10':
        args.mean_norm, args.std_norm = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        args.ncls, args.input_size = 10, 32
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm, args.std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        args.ncls, args.input_size = 200, 64
    elif args.dataset_name == 'SC_imsize64':
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

    # Initialize CutMix augmentation
    args.mixup_fn = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.0, label_smoothing=0.1,
                          num_classes=args.ncls, )
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def generate_models_process(rank, device, args):
    # set up for Multi-gpu training
    if args.use_multi_gpu:
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        device = device + f":{rank}"
    else:
        print(f"Using {device} for training")
    final_model_path = os.path.join(args.save_dir, f"{args.exp_name}.pth")
    trainloader, testloader = load_dataset(rank, args)
    loss_container_cal, raw_metric_cal, crop_metric_cal, drop_metric_cal = None, None, None, None
    loss_container_backbone, raw_metric_backbone = None, None
    if rank == 0:
        loss_container_cal, raw_metric_cal = AverageMeter(name='loss'), TopKAccuracyMetric(topk=(1, 5))
        crop_metric_cal, drop_metric_cal = TopKAccuracyMetric(topk=(1, 5)), TopKAccuracyMetric(topk=(1, 5))
        loss_container_backbone, raw_metric_backbone = AverageMeter(name='loss'), TopKAccuracyMetric(topk=(1, 5))

    for model_id, model_name in enumerate(args.model_list):
        if rank == 0:
            print(f"Start training model: {model_name}")
            if args.matplotlib:  # Visualization
                os.makedirs(os.path.join("results", args.dataset_name, args.model_source[model_id]), exist_ok=True)
            print(f"args:\n{args}")

        model = load_model(model_name, args.ncls, args.model_source[model_id], args.pretrained_weights,
                           args.pretrained_bn).to(device)
        if args.load_uncal_weights:
            state_dict = torch.load(str(os.path.join(args.save_dir, f"{model_name}.pth")), weights_only=True)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                print(f"{args.model_source[model_id]}'s {model_name} can't match ckpt")
            else:
                print(f"{args.model_source[model_id]}'s {model_name} can match ckpt")
        module = get_module(model_name, args.model_source[model_id], model)
        last_feature_hook = LastFeatureHook(module)
        cal = CAL(num_classes=args.ncls, M=args.M, net=model_name, source=args.model_source[model_id]).to(device)
        feature_center = torch.zeros(args.ncls, cal.M * cal.num_features).to(device)  # [num_classes, M * C]
        if args.use_multi_gpu:
            model = DDP(model, device_ids=[rank, ], output_device=rank)
            cal = DDP(cal, device_ids=[rank, ], output_device=rank)
        # setup loss function and optimizer
        criterion, center_loss = nn.CrossEntropyLoss().to(device), CenterLoss().to(device)
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(list(model.parameters()) + list(cal.parameters()), lr=args.lr[model_id],
                                        momentum=0.9, weight_decay=1e-5)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(list(model.parameters()) + list(cal.parameters()), lr=args.lr[model_id],
                                         weight_decay=1e-5)
        else:
            raise ValueError("Now only SGD and Adam")
        if args.cos_lr:
            scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)  # eta_min=1e-5
        else:
            scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                             lambda step: args.rate ** (step / args.duration),
                                                             last_epoch=-1)

        epochs, train_loss_list_cal, train_loss_list_backbone, test_loss_list_cal, lr_list = [], [], [], [], []
        test_loss_list_backbone, train_acc_list_backbone, test_acc_list_backbone = [], [], []
        train_acc_list_cal, train_acc_aux_list_cal, test_acc_list_cal, test_acc_aux_list_cal = [], [], [], []
        best_acc, best_epoch, best_backbone_acc, best_cal_acc = 0., 0, 0., 0.
        # train the model
        for epoch in range(0, args.epoch):
            if rank == 0:
                loss_container_cal.reset()
                raw_metric_cal.reset()
                crop_metric_cal.reset()
                drop_metric_cal.reset()
                loss_container_backbone.reset()

            model.train()
            cal.train()
            if args.use_multi_gpu:
                trainloader.sampler.set_epoch(epoch)
            # Train the model for one step
            for i, (inputs, labels) in enumerate(trainloader):
                # inputs, labels = args.mixup_fn(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_backbone = (1.0 - args.cal_ratio) * criterion(outputs, labels)
                last_feature = last_feature_hook.feature
                pred_raw, pred_eff, feature_matrix, attention_map, attention_maps = cal(last_feature)
                # Update Feature Center
                feature_center_batch = F.normalize(feature_center[labels], dim=-1)  # [B, M * C]
                delta_full = torch.zeros_like(feature_center, device=device)  # [num_classes, M * C]
                delta_full[labels] = feature_matrix.detach() - feature_center_batch
                #     local classes count
                counts_local = torch.zeros(args.ncls, device=device)
                ones = torch.ones_like(labels, device=device, dtype=counts_local.dtype)
                counts_local.scatter_add(0, labels, ones)
                #     global communication
                if args.use_multi_gpu:
                    dist.all_reduce(counts_local, op=dist.ReduceOp.SUM)
                    dist.all_reduce(delta_full, op=dist.ReduceOp.SUM)
                counts_global = counts_local.unsqueeze(1)
                mask = counts_global == 0
                counts_global[mask] = 1.0
                feature_center += 5e-2 * (delta_full / counts_global)
                # Attention Cropping
                with torch.no_grad():
                    crop_images = batch_augment(inputs, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                    drop_images = batch_augment(inputs, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
                aug_images = torch.cat([crop_images, drop_images], dim=0)  # [B * 2, 3, H ,W]
                labels_aug = torch.cat([labels, labels], dim=0)  # [B * 2]
                outputs_aug = model(aug_images)
                last_feature_aug = last_feature_hook.feature
                # [B * 2, num_classes], [B * 2, num_classes], [B * 2, M, C], [B * 2, M, AH, AW], crop images forward
                pred_raw_aug, pred_eff_aug, feature_matrix_aug, attention_map_aug, attention_maps_aug = cal(
                    last_feature_aug)
                pred_eff_aux = torch.cat([pred_eff, pred_eff_aug], dim=0)  # [B + B * 2, num_classes]
                labels_aux = torch.cat([labels, labels_aug], dim=0)  # [B + B * 2]
                # loss
                loss_cal = args.cal_ratio * (criterion(pred_raw, labels) / 3. +
                                             criterion(pred_eff_aux, labels_aux) * 3. / 3. +
                                             criterion(pred_raw_aug, labels_aug) * 2. / 3. +
                                             center_loss(feature_matrix, feature_center_batch))
                loss = loss_backbone + loss_cal
                loss.backward()
                optimizer.step()

                if rank == 0:
                    with torch.no_grad():
                        epoch_loss_cal = loss_container_cal(loss_cal.item())
                        epoch_raw_acc_cal = raw_metric_cal(pred_raw, labels)
                        epoch_crop_acc_cal = crop_metric_cal(pred_raw_aug, labels_aug)
                        epoch_drop_acc_cal = drop_metric_cal(pred_eff_aux, labels_aux)
                        epoch_loss_backbone = loss_container_backbone(loss_backbone.item())
                        epoch_raw_acc_backbone = raw_metric_backbone(outputs, labels)
                    if i == len(trainloader) - 1:
                        print(f"Train Epoch:{epoch}\n"
                              f"Loss:cal-{epoch_loss_cal:.4f},{model_name}-{epoch_loss_backbone:.4f}\n"
                              f"Acc:Raw:cal-{epoch_raw_acc_cal[0]:.2f},{model_name}-{epoch_raw_acc_backbone[0]:.2f};"
                              f"Crop:cal-{epoch_crop_acc_cal[0]:.2f};"
                              f"Drop:cal-{epoch_drop_acc_cal[0]:.2f}")

            if rank == 0 and (epoch % 10 == 0 or epoch == args.epoch - 1):
                print(f"Epoch:{epoch}")
                t_loss_cal, t_loss_backbone, t_acc_cal, t_acc_backbone, t_acc_aux, t_time \
                    = evaluate_loader_cal(model, cal, trainloader, device, last_feature_hook, loss_container_cal,
                                          raw_metric_cal, drop_metric_cal, loss_container_backbone, raw_metric_backbone,
                                          criterion)
                print(f"Train:\n"
                      f"Loss:cal-{t_loss_cal:.4f},{model_name}-{t_loss_backbone:.4f}\n"
                      f"Acc:Raw:cal-{t_acc_cal[0]:.2f},{model_name}-{t_acc_backbone[0]:.2f};"
                      f"Aux:cal-{t_acc_aux[0]:.2f};"
                      f"Time:{t_time}")
                v_loss_cal, v_loss_backbone, v_acc_cal, v_acc_backbone, v_acc_aux, v_time \
                    = evaluate_loader_cal(model, cal, testloader, device, last_feature_hook, loss_container_cal,
                                          raw_metric_cal, drop_metric_cal, loss_container_backbone, raw_metric_backbone,
                                          criterion)
                print(f"Test:\n"
                      f"Loss:cal-{v_loss_cal:.4f},{model_name}-{v_loss_backbone:.4f}\n"
                      f"Acc:Raw:cal-{v_acc_cal[0]:.2f},{model_name}-{v_acc_backbone[0]:.2f};"
                      f"Aux:cal-{v_acc_aux[0]:.2f};"
                      f"Time:{v_time}")
                if args.matplotlib:
                    # epoch
                    epochs.append(epoch)
                    # train loss
                    train_loss_list_cal.append(t_loss_cal)
                    train_loss_list_backbone.append(t_loss_backbone)
                    # train acc
                    train_acc_list_cal.append(t_acc_cal[0])
                    train_acc_aux_list_cal.append(t_acc_aux[0])
                    train_acc_list_backbone.append(t_acc_backbone[0])
                    # test loss
                    test_loss_list_cal.append(v_loss_cal)
                    test_loss_list_backbone.append(v_loss_backbone)
                    # test acc
                    test_acc_list_cal.append(v_acc_cal[0])
                    test_acc_aux_list_cal.append(v_acc_aux[0])
                    test_acc_list_backbone.append(v_acc_backbone[0])
                    # lr
                    lr_list.append(optimizer.param_groups[0]["lr"])
                    draw(args.exp_name, epochs, train_loss_list_cal, train_acc_list_cal, train_acc_aux_list_cal,
                         test_loss_list_cal, test_acc_list_cal, test_acc_aux_list_cal, train_loss_list_backbone,
                         train_acc_list_backbone, test_loss_list_backbone, test_acc_list_backbone, lr_list,
                         args.dataset_name, args.model_source[model_id], args.cal_ratio, model_name)

                now_acc = {f"{model_name}": v_acc_backbone[0], "cal": v_acc_cal[0]}
                if max(now_acc[f"{model_name}"], now_acc["cal"]) > best_acc:
                    if args.use_multi_gpu:
                        torch.save({model_name: model.module.state_dict(), "cal": cal.module.state_dict(),
                                    "feature_center": feature_center.cpu()}, final_model_path)
                    else:
                        torch.save({model_name: model.state_dict(), "cal": cal.state_dict(),
                                    "feature_center": feature_center.cpu()}, final_model_path)
                    best_acc, best_epoch = max(now_acc[f"{model_name}"], now_acc["cal"]), epoch
                    best_backbone_acc, best_cal_acc = now_acc[f"{model_name}"], now_acc["cal"]
                print(f"{model_name}:{best_backbone_acc:.4f},cal:{best_cal_acc:.4f}, epoch {best_epoch} were saved!")
            scheduler_lr.step()
            if epoch == args.stop_epoch: break


    if args.use_multi_gpu:
        dist.destroy_process_group()


def main_generate_pools(args):
    # Generating Pools for different Models, case when using more than 1 gpu
    if args.use_multi_gpu:
        if torch.cuda.device_count() < 2:
            print("The number of available gpus is less than 2, please use normal mode ")
            sys.exit()
        if args.world_size > torch.cuda.device_count() or args.world_size == -1:
            print(f"please set world size below the number of current available gpus: {torch.cuda.device_count()} ")
            sys.exit()
        setup_distributed_environment(master_port=args.master_port)
        print("Using Multi GPU Training....")
        mp.spawn(generate_models_process, args=("cuda", args), nprocs=args.world_size, join=True)
    # case when using one gpu or use cpu or use mps
    else:
        # setup device for training
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        generate_models_process(0, device, args)

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()
    # Step 1: main entry to generate pools
    main_generate_pools(args)
