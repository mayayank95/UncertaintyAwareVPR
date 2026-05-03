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
    """Initialize variance head bias so activation(0) ≈ target_variance.
    Weights keep default (Kaiming) for input sensitivity."""
    if activation == "softplus":
        bias_val = math.log(math.exp(target_variance) - 1)
    else:  # sigmoid
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

    backbone_name = opt.get("backbone", "ResNet18")
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    if var_type == "activation":
        return nn.Sequential(activation), False

    needs_feature_map = False

    if var_type == "linear":
        head = nn.Sequential(
            nn.Linear(features_dim, fc_output_dim),
            activation,
        )
    elif var_type == "mlp":
        
        head = nn.Sequential(
            GeM(),
            Flatten(),
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, fc_output_dim),
            activation,
        )
        needs_feature_map = True
    elif var_type == "separate_agg":
        assert aggregation is not None, "separate_agg requires the aggregation module"
        agg_copy = copy.deepcopy(aggregation)
        head = nn.Sequential(agg_copy, activation)
        needs_feature_map = True
    elif var_type == "separate_linear_agg":
        assert aggregation is not None, "separate_agg requires the aggregation module"
        agg_copy = copy.deepcopy(aggregation)
        head = nn.Sequential(agg_copy,nn.Linear(features_dim, fc_output_dim), activation)
        needs_feature_map = True
    elif var_type == "vmf":
        head = nn.Sequential(
            nn.Linear(fc_output_dim, 1),
            activation,
        )
    elif var_type == "vmf_agg":
        assert aggregation is not None, "vmf_agg requires the aggregation module"
        agg_copy = copy.deepcopy(aggregation)
        # The aggregation produces a descriptor of size fc_output_dim; we then predict kappa in [1] via Linear.
        head = nn.Sequential(
            agg_copy,
            nn.Linear(fc_output_dim, 1),
            activation,
        )
        needs_feature_map = True
    elif var_type == "stun_head":
        sigma_dim = opt.get("sigma_dim") or fc_output_dim
        head = nn.Sequential(
            nn.Linear(fc_output_dim, sigma_dim),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unknown var_head_type: {var_type}")

    if opt.get("var_init"):
        _stable_var_init(head, act_name)
    return head, needs_feature_map


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
        # No torch.no_grad(): variance path must retain grad for uncertainty loss backward.
        x = self.backbone(inputs)

        # Mean path
        desc = self.aggregation(x)
        mu = self.final_l2(desc)

        # Variance path (gradients flow through var_head when loss.backward() is called)
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
    x = torch.rand((4, 3, 224, 224))  # small batch to check variance stats

    basic = Basic(default_opt)
    print("Basic:", basic(x)[0].shape, basic(x)[1].shape)

    # 1) Init check: var_init=True => mean variance ~0.1 for linear, mlp, separate_agg
    print("\n--- Variance head init (var_init=True => mean variance ~0.1) ---")
    for vtype in ("activation", "linear", "mlp", "separate_agg", "vmf_agg"):
        # activation has no trainable params so var_init is N/A; others use var_init=True
        use_init = vtype != "activation"
        opt = {**default_opt, "var_head_type": vtype, "var_init": use_init}
        unc = Uncertainty(opt)
        unc.eval()
        with torch.no_grad():
            mu, var = unc(x)
        mean_var = var.mean().item()
        trainable = sum(p.numel() for p in unc.parameters() if p.requires_grad)
        expect = "~0.1" if use_init else "N/A (no init)"
        print(f"  {vtype:14s} var_init={use_init!s:5s}  mean(variance)={mean_var:.4f}  (expect {expect})  trainable={trainable}")

    # 2) Check that variance updates over training (var head receives gradients)
    n_steps = 100
    print(f"\n--- Variance updates over training ({n_steps} steps) ---")
    for vtype in ("linear", "mlp", "separate_agg"):
        opt = {**default_opt, "var_head_type": vtype, "var_init": True}
        model = Uncertainty(opt)
        if hasattr(model, "freeze_base"):
            model.freeze_base()  # train only var head
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        criterion = nn.GaussianNLLLoss()
        x_b = torch.randn(4, 3, 224, 224)
        model.train()
        with torch.no_grad():
            var0 = model(x_b)[1]
        mean_before = var0.mean().item()
        mean_at_20 = None
        for step in range(n_steps):
            optimizer.zero_grad()
            mu, var = model(x_b)
            scale = math.sqrt(mu.shape[1])
            target = mu.detach() + 0.1 * torch.randn_like(mu)
            loss = criterion(mu * scale, target * scale, var)
            loss.backward()
            optimizer.step()
            if step == 19:
                with torch.no_grad():
                    mean_at_20 = model(x_b)[1].mean().item()
        with torch.no_grad():
            mean_after = model(x_b)[1].mean().item()
        changed = abs(mean_after - mean_before) > 0.001
        print(f"  {vtype:14s}  before={mean_before:.4f}  after 20={mean_at_20:.4f}  after {n_steps}={mean_after:.4f}  updated={changed}")
    print("Done.")
