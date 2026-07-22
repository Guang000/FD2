import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        elif pool == "GMP":
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        AB, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))
        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)
        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-6)
        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + 1e-6)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class CAL(nn.Module):
    def __init__(self, num_classes, M=32, net='ResNet18', source="torchvision"):
        super(CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        # backbone's last features' shape when input 1 image
        if 'ResNet18' == net:
            if source == 'torchvision':
                self.shape_features = (512, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (512, 28, 28)
        elif 'ResNet50' == net:
            if source == 'torchvision':
                self.shape_features = (2048, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (2048, 28, 28)
        elif 'ResNet101' == net:
            if source == 'torchvision':
                self.shape_features = (2048, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (2048, 28, 28)
        elif 'Densenet121' == net:
            if source == 'torchvision':
                self.shape_features = (1024, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (1024, 28, 28)
        elif 'Densenet169' == net:
            if source == 'torchvision':
                self.shape_features = (1664, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (1664, 28, 28)
        elif 'Densenet201' == net:
            if source == 'torchvision':
                self.shape_features = (1920, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (1920, 28, 28)
        elif 'Densenet161' == net:
            if source == 'torchvision':
                self.shape_features = (2208, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (2208, 28, 28)
        elif 'MobileNetV2' == net:
            if source == 'torchvision':
                self.shape_features = (1280, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (1280, 28, 28)
        elif 'ShuffleNetV2' == net:
            if source == 'torchvision':
                self.shape_features = (1024, 7, 7)
            elif source == 'CVDD':
                self.shape_features = (1024, 28, 28)
        else:
            raise ValueError('Unsupported net: %s' % net)
        self.num_features = self.shape_features[0]
        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

    def visualize(self, feature_maps):
        # Attention Maps and Feature Matrix
        attention_maps = self.attentions(feature_maps)
        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)
        return p, attention_maps

    def forward(self, feature_maps):
        # Attention Maps and Feature Matrix
        batch_size = feature_maps.shape[0]
        attention_maps = self.attentions(feature_maps)  # [B, M, AH, AW]
        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)  # [B, M * C], [B, M * C]
        # Classification
        p = self.fc(feature_matrix * 100.)  # [B, num_classes]
        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + 1e-6)  # [M]
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])  # [2, AH, AW]
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
        # [B, num_classes], [B, num_classes],               [B, M * C],  [B, 2/1, H, W]
        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map, attention_maps

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(CAL, self).load_state_dict(model_dict)