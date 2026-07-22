import torch
import argparse
from utils_squeeze import load_dataset, evaluate_loader, get_all_models
from models.utils_models import load_model
import torch.nn as nn
import csv
import matplotlib

matplotlib.use('Agg')  # headless service
import matplotlib.pyplot as plt
import sys
import os
# for multiprocessing system
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.data import Mixup


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
    parser.add_argument('--pretrained_bn', action='store_true', help='whether to use pretrained bn')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataset_dir', type=str, required=True, help='directory where the dataset are stored')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the trained models')
    parser.add_argument('--batch_size', default=128, type=int, help='number of images to optimize at the same time')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset to use for training')
    parser.add_argument('--epoch', type=int, default=200, help='num of iterations to optimize the target model')
    parser.add_argument('--lr', nargs='+', type=float, help='learning rate for optimization')
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

    trainloader, testloader = load_dataset(rank, args)
    if rank == 0:  # Visualization
        if args.matplotlib:
            for i in range(len(args.model_source)):
                os.makedirs(os.path.join("results", args.dataset_name, args.model_source[i]), exist_ok=True)
    for model_id, model_name in enumerate(args.model_list):
        run = None
        if rank == 0:
            print(f"Start training model: {model_name}")
            if args.matplotlib:
                os.makedirs(os.path.join("results", args.dataset_name, args.model_source[model_id], model_name),
                            exist_ok=True)
            print(f"args:\n{args}")

        model = load_model(model_name, args.ncls, args.model_source[model_id], False, args.pretrained_bn).to(device)
        if args.use_multi_gpu: model = DDP(model, device_ids=[rank, ], output_device=rank)
        # setup loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr[model_id], momentum=0.9, weight_decay=1e-4)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr[model_id], weight_decay=1e-4)
        else:
            raise ValueError("Now only SGD and Adam")
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
        epochs, train_acc_list, train_loss_list, test_acc_list, test_loss_list, lr_list = [], [], [], [], [], []
        # train the model
        for epoch in range(0, args.epoch):
            model.train()
            if args.use_multi_gpu:
                trainloader.sampler.set_epoch(epoch)
            # Train the model for one step
            for inputs, labels in trainloader:
                # inputs, labels = args.mixup_fn(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # evaluate the model only in main gpu
            if rank == 0 and (epoch % 10 == 0 or epoch == args.epoch - 1):
                train_acc, train_loss = evaluate_loader(model, criterion, trainloader, device)
                test_acc, test_loss = evaluate_loader(model, criterion, testloader, device)
                print(f"Epoch:{epoch} Acc: train:{train_acc} test:{test_acc} Loss: train:{train_loss} test:{test_loss}")
                if args.matplotlib:
                    epochs.append(epoch)
                    # Draw Accuracy
                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    plt.figure(figsize=(6, 4))
                    plt.plot(epochs, train_acc_list, label='Acc/train', marker=",", linestyle="-", color="blue",
                             linewidth=0.5)
                    plt.plot(epochs, test_acc_list, label='Acc/test', marker=",", linestyle="-", color="green",
                             linewidth=0.5)
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.title("Accuracy vs. Epoch")
                    plt.legend()
                    plt.grid(False)
                    plt.savefig(os.path.join("results", args.dataset_name, args.model_source[model_id], model_name,
                                             "Accuracy.png"))
                    plt.close()
                    # Draw Loss
                    train_loss_list.append(train_loss)
                    test_loss_list.append(test_loss)
                    plt.figure(figsize=(6, 4))
                    plt.plot(epochs, train_loss_list, label='Loss/train', marker=",", linestyle="-", color="blue",
                             linewidth=0.5)
                    plt.plot(epochs, test_acc_list, label='Loss/test', marker=",", linestyle="-", color="green",
                             linewidth=0.5)
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Loss vs. Epoch")
                    plt.legend()
                    plt.grid(False)
                    plt.savefig(
                        os.path.join("results", args.dataset_name, args.model_source[model_id], model_name, "Loss.png"))
                    plt.close()
                    # Draw lr
                    lr_list.append(optimizer.param_groups[0]["lr"])
                    plt.figure(figsize=(6, 4))
                    plt.plot(epochs, lr_list, label='lr', marker=",", linestyle="-", color="purple", linewidth=0.5)
                    plt.xlabel("Epoch")
                    plt.ylabel("lr")
                    plt.title("lr vs. Epoch")
                    plt.legend()
                    plt.grid(False)
                    plt.savefig(
                        os.path.join("results", args.dataset_name, args.model_source[model_id], model_name, "lr.png"))
                    plt.close()

            scheduler_lr.step()

        # rank 0 save the model
        if rank == 0:
            final_model_path = os.path.join(args.save_dir, f"{model_name}.pth")
            if args.use_multi_gpu:
                torch.save(model.module.state_dict(), final_model_path)
            else:
                torch.save(model.state_dict(), final_model_path)
            print("finished processing model: ", model_name)


    if args.use_multi_gpu:
        dist.destroy_process_group()


def main_generate_pools(args):
    # Generating Pools for different Models, case when using more than 1 gpu
    if args.use_multi_gpu:
        if torch.cuda.device_count() < 2:
            print("The number of availabel gpus is less than 2, please use normal mode ")
            sys.exit()
        if args.world_size > torch.cuda.device_count() or args.world_size == -1:
            print(f"please set world size below the number of current availabele gpus: {torch.cuda.device_count()} ")
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
    args = parse_args()
    main_generate_pools(args)

