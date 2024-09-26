import torch
from torch import nn
from torchvision.models import (
    resnet18, resnet34, resnet50, 
    densenet121, densenet161, 
)

BACKBONE_MODELS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50' : resnet50,
    'densenet121': densenet121, 
    'densenet161': densenet161,
}


def get_backbone(net_name, num_classes, weights='DEFAULT'):
    """
    Returns a pretrained pytorch model based on the provided backbone name.
    
    Args:
        * net_name (string): Name of the network (e.g. resnet18 etc.)
        * num_classes (int): Number of output features
        * weights: ('DEFAULT' or None): Loads either default weights or no weights.
    """
    
    model_fn = BACKBONE_MODELS.get(net_name)
    if model_fn is None:
        raise KeyError(f'{net_name} not in available backbone models: ' \
                       + f'{", ".join(BACKBONE_MODELS.keys())}')
    

    backbone = model_fn(weights=weights)

    # Adjust the classifier or fully connected layer based on the model architecture
    if hasattr(backbone, 'classifier'):  # For models like DenseNet
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(num_features, num_classes)
    elif hasattr(backbone, 'fc'):  # For models like ResNet
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, num_classes)
    else:
        raise AttributeError('The provided model does not have a recognized classifier or fc layer.')
    
    return backbone



class Backbone(nn.Module):
    """
    Args:
        * net_name (string): Name of the network (e.g. resnet18 etc.)
        * num_classes (int): Number of output features
        * weights: ('DEFAULT' or None): Loads either default weights or no weights.
    """
    def __init__(self, net_name, num_classes, weights='DEFAULT'):
        super(Backbone, self).__init__()
        
        model_fn = BACKBONE_MODELS.get(net_name)
        if model_fn is None:
            raise KeyError(f'{net_name} not in available backbone models: ' \
                + f'{", ".join(BACKBONE_MODELS.keys())}')
        
        backbone = model_fn(weights=weights)
        

        # Adjust the classifier or fully connected layer based on the model architecture
        if hasattr(backbone, 'classifier'):  # For models like DenseNet
            num_features = backbone.classifier.in_features
            backbone.classifier = nn.Linear(num_features, 128 * 8 * 8)
        elif hasattr(backbone, 'fc'):  # For models like ResNet
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 128 * 8 * 8)
        else:
            raise AttributeError('The provided model does not have a recognized classifier or fc layer.')
        
        self.encoder = backbone
        self.classify = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, input):
        batch_size = len(input)
        x = self.encoder(input)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)
        return outLinear
