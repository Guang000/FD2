import os
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F
from models import *
from models.utils_models import load_model


class SimpleImageFolder(Dataset):
    def __init__(self, root, ipc, mode='train', memory=False, transform=None):
        self.root = os.path.join(root, mode)
        self.ipc = ipc
        self.memory = memory
        self.transform = transform
        self.loader = default_loader
        classes = sorted([cls for cls in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, cls))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.image_paths = [] 
        self.targets = []  
        self.samples = []  
        self._load_images()

    def _load_images(self):
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.root, cls_name)
            imgs_name = sorted([img_name for img_name in os.listdir(cls_dir)
                                if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for img_name in imgs_name[:self.ipc]:
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.targets.append(cls_idx)
                if self.memory:
                    self.samples.append(self.loader(img_path))

    def __getitem__(self, index):
        if self.memory:
            img = self.samples[index]
        else:
            img = self.loader(self.image_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.targets)


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=64, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(self.size // self.factor, ratio=(1, 1), antialias=True, )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


def pad(input_tensor, target_height, target_width=None):
    """
    Pad input tensor(shape=[batch_size, C, H, W]) to padded_tensor(shape=[batch_size, C, target_height, target_width])
    Args:
        input_tensor: shape=[batch_size, C, H, W]
        target_height: target height
        target_width: target width
    Returns: padded_tensor(shape=[batch_size, C, target_height, target_width]
    """
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)  # target_height-H
    horizontal_padding = target_width - input_tensor.size(3)  # target_width-W

    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding
    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding

    padded_tensor = F.pad(input_tensor, (left_padding, right_padding, top_padding, bottom_padding))

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)  # ipc * num_crop
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i: min(i + batch_size, total_samples)]
            output = model(batch_data)
            all_outputs.append(output)
    final_output = torch.cat(all_outputs, dim=0)
    return final_output


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def selector(best_crop_num, model, images, labels, size, device="cuda"):
    with torch.no_grad():
        images = images.to(device)
        s = images.shape  # [ipc, num_crop, 3, H, W]
        if best_crop_num > s[0]:
            raise ValueError(f"best_crop_num({best_crop_num}) can't be greater than ipc")
        images = images.permute(1, 0, 2, 3, 4)  # [num_crop, ipc, 3, H, W]
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])  # [num_crop * ipc, 3, H, W]
        labels = labels.repeat(s[1]).to(device)  # [ipc * num_crop]

        preds = batched_forward(model, pad(images, size).to(device), batch_size=s[0])  # [num_crop * ipc, num_class]

        # dist = cross_entropy(preds, labels)  # [num_crop * ipc]
        dist = F.cross_entropy(preds, labels, reduction='none')  # [num_crop * ipc]

        dist = dist.reshape(s[1], s[0])  # [num_crop, ipc]

        index = torch.argmin(dist, 0)  # [ipc]

        dist = dist[index, torch.arange(s[0])]  # [ipc]

        images = images.reshape(s[1], s[0], s[2], s[3], s[4])
        images = images[index, torch.arange(s[0])]  # [ipc, 3, H, W]

    indices = torch.argsort(dist, descending=False)[:best_crop_num]
    torch.cuda.empty_cache()
    return images[indices].detach()


def mix_images(input_img, out_size, factor, mixed_img_num):
    patch_size = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros((mixed_img_num, 3, out_size, out_size), requires_grad=False, dtype=torch.float, )
    h_loc = 0
    for i in range(factor):
        h_r = patch_size + 1 if i < remained else patch_size
        w_loc = 0
        for j in range(factor):
            w_r = patch_size + 1 if j < remained else patch_size
            img_part = F.interpolate(input_img.data[k * mixed_img_num: (k + 1) * mixed_img_num], size=(h_r, w_r))
            mixed_images.data[0:mixed_img_num, :, h_loc: h_loc + h_r, w_loc: w_loc + w_r,] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images


def save_images(root, images, class_id, img_id):
    dir_path = os.path.join(root, "{:05d}".format(class_id))
    os.makedirs(dir_path, exist_ok=True)
    place_to_store = os.path.join(dir_path, "class{:05d}_id{:05d}.jpg".format(class_id, img_id))
    image_np = images[0].data.cpu().numpy().transpose((1, 2, 0))
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    pil_image.save(place_to_store)


def make_patch(model_name, ckpt_path, ncls, src_dir, ipc, mean_norm, std_norm, patch_num, num_crop, imsize, save_dir):
    state_dict = torch.load(ckpt_path, weights_only=True)
    model = None
    try:
        model = load_model(model_name, ncls, "CVDD", True, True)
        model.load_state_dict(state_dict)
    except RuntimeError:
        print(f"CVDD's {model_name} can't match ckpt, next try torchvision")
        try:
            model = load_model(model_name, ncls, "torchvision")
            model.load_state_dict(state_dict)
        except BaseException:
            print(f"torchvision's {model_name} also can't match ckpt")
        else:
            print(f"torchvision's {model_name} can match ckpt")
    else:
        print(f"CVDD's {model_name} can match ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    trainset = SimpleImageFolder(src_dir, ipc=ipc, mode='train', memory=True, transform=None)

    trainset.transform = transforms.Compose([
        transforms.ToTensor(),
        MultiRandomCrop(num_crop=num_crop, size=imsize, factor=2),
        # transforms.Resize((64, 64)),
        transforms.Normalize(mean=mean_norm, std=std_norm),
    ])
    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / std_norm[0], 1 / std_norm[1], 1 / std_norm[2]]),
         transforms.Normalize(mean=[-mean_norm[0], -mean_norm[1], -mean_norm[2]], std=[1.0, 1.0, 1.0])])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=ipc, shuffle=False, num_workers=0, pin_memory=False)
    os.makedirs(save_dir, exist_ok=True)
    for img_id in range(patch_num):
        for c, (images, labels) in enumerate(train_loader):
            print(f"current img_id:{img_id}, class:{c}")
            images = selector(4, model, images, labels, size=imsize, device=device)
            images = mix_images(images, imsize, factor=2, mixed_img_num=1)
            save_images(save_dir, denormalize(images), c, img_id)


if __name__ == '__main__':
    model_name = "ResNet18"
    ncls = 200
    dataset_name = "CUB_imsize224"
    ipc = 29  # Minimum Number of Images in Each Class
    mean_norm = [0.4857, 0.4994, 0.4326]
    std_norm = [0.2260, 0.2215, 0.2595]
    imsize = 224
    patch_num = 5
    num_crop = 5
    src_dir = os.path.join("..", "Datasets", dataset_name)  # Replace with your path
    save_dir = os.path.join("..", "Datasets", "patches", dataset_name, "2")  # Replace with your path
    ckpt_path = os.path.join("..", "Datasets", "pretrained_models", dataset_name, model_name + ".pth")  # Replace with your path
    make_patch(model_name, ckpt_path, ncls, src_dir, ipc, mean_norm, std_norm, patch_num, num_crop, imsize, save_dir)
