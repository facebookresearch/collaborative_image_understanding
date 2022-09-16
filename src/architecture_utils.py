import torch.nn as nn
import torchvision.models as models


def get_backbone(is_pretrained: bool, arch: str = "resnet18"):
    # Define the backbone
    if arch == "resnet18":
        backbone = models.resnet18(pretrained=is_pretrained)
        out_feature_num = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

    elif arch == "mobilenet":
        backbone = models.mobilenet_v2(pretrained=is_pretrained)
        out_feature_num = backbone.classifier[-1].in_features
        layers = list(backbone.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))]

    elif arch == "regnet":
        backbone = models.regnet_y_400mf(pretrained=is_pretrained) # models.squeezenet1_1(pretrained=is_pretrained)
        out_feature_num = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
    else:
        raise ValueError(f"{arch=}")
    backbone = nn.Sequential(*layers)
    return backbone, out_feature_num


def get_classifier(in_features: int, num_target_classes: int):
    classifier = nn.Linear(in_features, num_target_classes)
    return classifier


def get_cf_predictor(num_filters: int, cf_vector_dim: int):
    # Define the cf vector predictor
    cf_layers = nn.Sequential(
        nn.BatchNorm1d(num_filters),
        nn.Linear(num_filters, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, cf_vector_dim),
    )
    return cf_layers
