import torch
import copy
import torch.nn as nn
from models.cosplace_uncertainty.cosplace_model.cosplace_network import GeoLocalizationNet
from models.cosplace_uncertainty.cosplace_model.layers import L2Norm


class GeneralModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output
        return output, torch.zeros_like(output)

class Basic(GeoLocalizationNet):
    def __init__(self, opt):
        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
        )
        self.id = 'basic'
        self.final_l2 = L2Norm()

    def forward(self, inputs):
        desc = super().forward(inputs)
        mu = self.final_l2(desc)
        return mu, torch.zeros_like(mu)


class Uncertainty(GeoLocalizationNet):
    def __init__(self, opt):
        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
        )
        self.id = 'uncertainty'

        self.separate_variance_aggregation = opt.get("separate_variance_aggregation", False)
        if self.separate_variance_aggregation:
            # Create a separate aggregation module for variance (copy of the mean aggregation)
            self.variance_aggregation = copy.deepcopy(self.aggregation)

        if opt.get("use_variance_linear", False):
            descriptors_dimension = opt.get("descriptors_dimension", 512)
            self.var_head = nn.Sequential(
                nn.Linear(descriptors_dimension, descriptors_dimension),
                nn.Softplus()
            )
        else:
            self.var_head = nn.Sequential(
                nn.Softplus()
            )
        self.final_l2 = L2Norm()

    def forward(self, inputs):
        if self.separate_variance_aggregation:
            x = self.backbone(inputs)

            # Mean path
            desc = self.aggregation(x)
            mu = self.final_l2(desc)

            # Variance path (separate aggregation)
            var_desc = self.variance_aggregation(x)
            log_sigma_sq = self.var_head(var_desc)

            return mu, log_sigma_sq
        else:
            # Shared path: backbone + aggregation -> desc
            desc = super().forward(inputs)       # [B, fc_out]
            mu = self.final_l2(desc)             # [B, fc_out], L2-normalized
            log_sigma_sq = self.var_head(desc)   # [B, sigma_dim], no L2
            return mu, log_sigma_sq

def deliver_model(opt):
    if opt['model_mode'] == 'basic':
        return Basic(opt)
    elif opt['model_mode'] == 'uncertainty':
        return Uncertainty(opt)


if __name__ == '__main__':
    default_opt = {"backbone": "ResNet18", "descriptors_dimension": 512}
    basic = Basic(default_opt)
    unc = Uncertainty(default_opt)
    x = torch.rand((1, 3, 224, 224))
    print("Basic:", basic(x)[0].shape, basic(x)[1].shape)
    print("Uncertainty:", unc(x)[0].shape, unc(x)[1].shape)
