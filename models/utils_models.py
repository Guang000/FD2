import torchvision
from models import *
import random
import timm

def get_bn_keys(model):
    bn_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            suffixes = ['weight', 'bias', 'running_mean', 'running_var']
            for suf in suffixes:
                key = f"{module_name}.{suf}" if module_name else suf
                bn_keys.add(key)
    return bn_keys


def load_state_dicts(net, state_dict, pretrained_bn=True):
    if pretrained_bn:
        net.load_state_dict(state_dict)
    else:
        bn_keys = get_bn_keys(net)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in bn_keys}
        net.load_state_dict(filtered_state_dict, strict=False)
    return net


def load_model(model, ncls, source="CVDD", pretrained_weights=True, pretrained_bn=True):
    if model == 'ResNet18':
        if source == 'CVDD':
            net = ResNet18(ncls)
        elif source == 'torchvision':
            net = torchvision.models.resnet18(weights=None)
            if pretrained_weights:
                state_dict = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict()
                net = load_state_dicts(net, state_dict, pretrained_bn)
            net.fc = nn.Linear(net.fc.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'ResNet50':
        if source == 'CVDD':
            net = ResNet50(ncls)
        elif source == 'torchvision':
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            if pretrained_weights:
                state_dict = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.get_state_dict()
                net = load_state_dicts(net, state_dict, pretrained_bn)
            net.fc = nn.Linear(net.fc.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'ResNet101':
        if source == 'CVDD':
            net = ResNet101(ncls)
        elif source == 'torchvision':
            net = torchvision.models.resnet101(weights=None)
            if pretrained_weights:
                state_dict = torchvision.models.ResNet101_Weights.IMAGENET1K_V1.get_state_dict()
                net = load_state_dicts(net, state_dict, pretrained_bn)
            net.fc = nn.Linear(net.fc.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'Densenet121':
        if source == 'CVDD':
            net = DenseNet121(ncls)
        elif source == 'torchvision':
            if pretrained_weights:
                net = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
                if not pretrained_bn:
                    for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
            else:
                net = torchvision.models.densenet121(weights=None)
            net.classifier = nn.Linear(net.classifier.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'Densenet161':
        if source == 'CVDD':
            net = DenseNet161(ncls)
        elif source == 'torchvision':
            if pretrained_weights:
                net = torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.IMAGENET1K_V1)
                if not pretrained_bn:
                    for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
            else:
                net = torchvision.models.densenet161(weights=None)
            net.classifier = nn.Linear(net.classifier.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'Densenet169':
        if source == 'CVDD':
            net = DenseNet169(ncls)
        elif source == 'torchvision':
            if pretrained_weights:
                net = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)
                if not pretrained_bn:
                    for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
            else:
                net = torchvision.models.densenet169(weights=None)
            net.classifier = nn.Linear(net.classifier.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'Densenet201':
        if source == 'CVDD':
            net = DenseNet201(ncls)
        elif source == 'torchvision':
            if pretrained_weights:
                net = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1)
                if not pretrained_bn:
                    for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
            else:
                net = torchvision.models.densenet201(weights=None)
            net.classifier = nn.Linear(net.classifier.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'MobileNetV2':
        if source == 'CVDD':
            net = MobileNetV2(ncls)
        elif source == 'torchvision':
            net = torchvision.models.mobilenet_v2(weights=None)
            if pretrained_weights:
                state_dict = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.get_state_dict()
                net = load_state_dicts(net, state_dict, pretrained_bn)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == 'ShuffleNetV2':
        if source == 'CVDD':
            net = ShuffleNetV2(net_size=0.5, ncls=ncls)
        elif source == 'torchvision':
            net = torchvision.models.shufflenet_v2_x0_5(weights=None)
            if pretrained_weights:
                state_dict = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.get_state_dict()
                net = load_state_dicts(net, state_dict, pretrained_bn)
            net.fc = nn.Linear(net.fc.in_features, ncls, bias=True)
        else:
            raise ValueError(f"Source {source} don't support {model}")
    elif model == "ViT":
        print(f"The model's source will be torchvision when using {model}")
        if pretrained_weights:
            net = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
            if not pretrained_bn:
                for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
        else:
            net = torchvision.models.vit_b_32(weights=None)
        net.heads.head = nn.Linear(net.heads.head.in_features, ncls, bias=True)
    elif model == "ConvNeXt":
        print(f"The model's source is torchvision when using {model}")
        if pretrained_weights:
            net = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            if not pretrained_bn:
                for m in net.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            m.reset_parameters()
        else:
            net = torchvision.models.convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, ncls)
    elif model == "DeiT":
        print(f"The model's source is timm when using {model}")
        net = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained_weights, num_classes=ncls)
    else:
        raise ValueError(f'Model {model} not supported')
    return net


def get_module(model_name, model_source, model):
    module = None
    if model_name == 'ResNet18':
        if model_source == "torchvision":
            module = model.layer4[1].bn2
        elif model_source == "CVDD":
            module = model.layer4[1].bn2
    elif model_name == 'ResNet50':
        if model_source == "torchvision":
            module = model.layer4[2].relu
        elif model_source == "CVDD":
            module = model.layer4[2].bn3
    elif model_name == 'ResNet101':
        if model_source == "torchvision":
            module = model.layer4[2].relu
        elif model_source == "CVDD":
            module = model.layer4[2].bn3
    elif model_name == 'Densenet121':
        if model_source == "torchvision":
            module = model.features.norm5
        elif model_source == "CVDD":
            module = model.bn
    elif model_name == 'Densenet169':
        if model_source == "torchvision":
            module = model.features.norm5
        elif model_source == "CVDD":
            module = model.bn
    elif model_name == 'Densenet201':
        if model_source == "torchvision":
            module = model.features.norm5
        elif model_source == "CVDD":
            module = model.bn
    elif model_name == 'Densenet161':
        if model_source == "torchvision":
            module = model.features.norm5
        elif model_source == "CVDD":
            module = model.bn
    elif model_name == 'MobileNetV2':
        if model_source == "torchvision":
            module = model.features[18][2]
        elif model_source == "CVDD":
            module = model.bn2
    elif model_name == 'ShuffleNetV2':
        if model_source == "torchvision":
            module = model.conv5[2]
        elif model_source == "CVDD":
            module = model.bn2
    return module


class LastFeatureHook:  # 注册前向传播钩子函数
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output

    def close(self):
        self.handle.remove()


class AverageMeter:
    def __init__(self, name='loss'):
        self.total_num = None
        self.scores = None
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric:
    def __init__(self, topk=(1,)):
        self.num_samples = None
        self.corrects = None
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, dim=1, largest=True, sorted=True)  # _, [maxk, B]
        pred = pred.t()  # [B, maxk]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, imgC, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            theta_c = random.uniform(*theta) * atten_map.max() if isinstance(theta, tuple) else theta * atten_map.max()
            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=False) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            crop_images.append(
                F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                              size=(imgH, imgW), mode='bilinear', align_corners=False))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            theta_d = random.uniform(*theta) * atten_map.max() if isinstance(theta, tuple) else theta * atten_map.max()
            drop_masks.append(
                F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=False) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError(f'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method {mode}')


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)