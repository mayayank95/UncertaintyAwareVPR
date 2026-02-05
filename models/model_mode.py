#%%
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, opt=None):
        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
            uncertainty_mode=False,   # Include L2Norm as the last layer
        )
        self.id = 'basic'

    def forward(self, inputs):
        mu = super().forward(inputs)   # כבר L2-normalized
        return mu, torch.zeros_like(mu)


class Uncertainty(GeoLocalizationNet):
    def __init__(self, opt=None):

        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
            uncertainty_mode=True,   # without l2 in the last layer
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
        self.final_l2 = L2Norm()  # L2 for mean

    def forward(self, inputs):
        if self.separate_variance_aggregation:
            x = self.backbone(inputs)
            
            # Mean path
            desc = self.aggregation(x)
            mu = self.final_l2(desc)

            # Variance path
            var_desc = self.variance_aggregation(x)
            log_sigma_sq = self.var_head(var_desc)
            
            return mu, log_sigma_sq
        else:
            # super().forward --> backbone + aggregation (GeM + Flatten + Linear), without L2
            desc = super().forward(inputs)      # shape: [B, fc_out]

            #  mean with L2
            mu = self.final_l2(desc)           # [B, fc_out]

            # variance without L2
            log_sigma_sq = self.var_head(desc) # [B, sigma_dim]

            return mu, log_sigma_sq

def deliver_model(opt):
    if opt['model_mode'] == 'basic':
        return Basic(opt)
    elif opt['model_mode'] == 'uncertainty':
        return Uncertainty(opt)


if __name__ == '__main__':
    tea = Basic()
    stu = Uncertainty()
    inputs = torch.rand((1, 3, 224, 224))
    outputs_tea = tea(inputs)
    outputs_stu = stu(inputs)

    print(outputs_tea[0].shape, outputs_tea[1].shape)
    print(outputs_stu[0].shape, outputs_stu[1].shape)
