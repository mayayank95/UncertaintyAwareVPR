import math
import torch
import copy
import torch.nn as nn
from models.cosplace_uncertainty.cosplace_model.cosplace_network import GeoLocalizationNet, CHANNELS_NUM_IN_LAST_CONV
from models.cosplace_uncertainty.cosplace_model.layers import L2Norm, GeM, Flatten


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


def _stable_var_init(module, activation="softplus", target_variance=0.1):
    """Initialize variance head bias so the initial output is centered at target_variance.
    Weights keep default Kaiming init for input sensitivity."""
    if activation == "softplus":
        bias_val = math.log(math.exp(target_variance) - 1)
    else:  # sigmoid: inverse sigmoid(target)
        bias_val = -math.log(1.0 / target_variance - 1)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def _build_var_head(opt, fc_output_dim, aggregation=None):
    """Build variance head module based on var_head_type flag.
    Returns (var_head, needs_feature_map): needs_feature_map is True when the head
    operates on the backbone feature map rather than the aggregated descriptor."""
    var_type = opt.get("var_head_type", "linear")
    act_name = opt.get("variance_activation", "softplus") or "softplus"
    activation = nn.Sigmoid() if act_name == "sigmoid" else nn.Softplus()

    if var_type == "activation":
        return nn.Sequential(activation), False

    if var_type == "linear":
        head = nn.Sequential(
            nn.Linear(fc_output_dim, fc_output_dim),
            activation,
        )
        if opt.get("var_init"):
            _stable_var_init(head, act_name)
        return head, False

    if var_type == "mlp":
        backbone_name = opt.get("backbone", "ResNet18")
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
        head = nn.Sequential(
            GeM(),
            Flatten(),
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, fc_output_dim),
            activation,
        )
        if opt.get("var_init"):
            _stable_var_init(head, act_name)
        return head, True

    if var_type == "separate_agg":
        assert aggregation is not None, "separate_agg requires the aggregation module"
        agg_copy = copy.deepcopy(aggregation)
        head = nn.Sequential(agg_copy, activation)
        if opt.get("var_init"):
            _stable_var_init(head, act_name)
        return head, True

    raise ValueError(f"Unknown var_head_type: {var_type}")


class Uncertainty(GeoLocalizationNet):
    def __init__(self, opt):
        super().__init__(
            backbone=opt.get("backbone", "ResNet18"),
            fc_output_dim=opt.get("descriptors_dimension", 512),
            train_all_layers=opt.get("train_all_layers", False),
        )
        self.id = 'uncertainty'
        fc_output_dim = opt.get("descriptors_dimension", 512)

        self.var_head, self._var_from_feature_map = _build_var_head(
            opt, fc_output_dim, aggregation=self.aggregation
        )
        self.final_l2 = L2Norm()

    def forward(self, inputs):
        x = self.backbone(inputs)

        # Mean path
        desc = self.aggregation(x)
        mu = self.final_l2(desc)

        # Variance path
        if self._var_from_feature_map:
            variance = self.var_head(x)
        else:
            variance = self.var_head(desc)

        return mu, variance + 1e-6

def deliver_model(opt):
    if opt['model_mode'] == 'basic':
        return Basic(opt)
    elif opt['model_mode'] == 'uncertainty':
        return Uncertainty(opt)


if __name__ == '__main__':
    default_opt = {"backbone": "ResNet18", "descriptors_dimension": 512}
    x = torch.rand((1, 3, 224, 224))

    basic = Basic(default_opt)
    print("Basic:", basic(x)[0].shape, basic(x)[1].shape)

    for vtype in ("activation", "linear", "mlp", "separate_agg"):
        opt = {**default_opt, "var_head_type": vtype, "var_init": vtype not in ("activation", "separate_agg")}
        unc = Uncertainty(opt)
        mu, var = unc(x)
        trainable = sum(p.numel() for p in unc.parameters() if p.requires_grad)
        print(f"Uncertainty ({vtype}): mu={mu.shape}, var={var.shape}, trainable={trainable}")
