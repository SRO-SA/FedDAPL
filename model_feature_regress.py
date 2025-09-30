""" 3D brain age model with separate feature extractor and regressor"""
# mean absolute error
# adam optimizer
# learning rate 10^-4
# weight decay 10^-4
from torch import nn
import torch
from utils import get_layer_params_list, get_layer_params_dict, flatten_layer_param_list_for_model
from utils import log_print
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from discriminator import DomainDiscriminator, GradientReverseLayer, ScannerBranch, IdentityScanner

def conv_blk(in_channel, out_channel, norm='instance'):
    layers = [nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)]
    if norm == 'batch':
        layers.append(nn.BatchNorm3d(out_channel))
    elif norm == 'none':
        pass
    else:
        layers.append(nn.InstanceNorm3d(out_channel))
    layers += [nn.MaxPool3d(2, stride=2), nn.ReLU()]
    return nn.Sequential(*layers)

    # return nn.Sequential(
    #     nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
    #     nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    # )



# ------------------ 1. Featurizer ------------------ #
class BrainCancerFeaturizer(nn.Module):
    """Return the feature *map* produced by conv4 or conv5
       + a global-pooled vector version (d_out = 256)."""
    def __init__(self, use_conv5: bool = True, channel_num: int = 1, d_out: int = 256):
        super().__init__()
        # copy/paste from your original model
        self.channel_num = channel_num
        self.conv1 = conv_blk(1, 32)  # input is (B,1,D,H,W)
        self.conv2 = conv_blk(32, 64)
        self.conv3 = conv_blk(64, 128)  # (B,128,D',H',W')
        self.conv4 = conv_blk(128, 256)  # (B,256,D'',H'',W'')
        self.conv5 = conv_blk(256, 256)           # ← optional
        self.use_conv5 = use_conv5
        # self.batch_norm = True
        # global pooling produces (B,256,1,1,1) → (B,256)
        # self.gap = nn.AdaptiveAvgPool3d(1)
        self.d_out = d_out *3*4*3                    # <- for DomainDiscriminator
        self.batch_norm = True                   # matches InstanceNorm

    def forward(self, x: torch.Tensor, tap='conv5') -> torch.Tensor:
        if torch.equal(torch.Tensor(list(x.shape)), torch.Tensor([1, 1, 1, 1, 121, 145, 121])):
            print("in if")
            x = torch.squeeze(x, dim = 0)
        x = x.reshape(-1, 1, *x.shape[-3:])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if tap == 'conv3':
            # print("x shape after conv3", x.shape)
            return x
        x = self.conv4(x)
        if tap == 'conv4':
            return x
        if self.use_conv5:
            x = self.conv5(x)                     # features after conv5
        # print("x shape after conv5", x.shape)
        return x                                  # (B,256,D',H',W')

    def map_to_vec(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.gap(fmap).flatten(1)          # (B,256)


# ------------------ 2. Regressor head ------------------ #
class BrainCancerRegressor(nn.Module):
    """Takes the feature map produced above and outputs the age scalar."""
    def __init__(self):
        super().__init__()
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        )
        self.drop   = nn.Identity()
        self.output = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        # nn.init.constant_(self.output.bias, 62.68)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = self.conv6(fmap)
        x = self.drop(x)
        x = self.output(x)
        x = x.flatten(1)
        return x            # (B,)  – regression value


# ------------------ 3. Full DANN wrapper ------------------ #
class DANN3D(nn.Module):
    """Domain-Adversarial Net for 3-D volumes – plugs straight into your trainer."""
    def __init__(self,
                 featurizer: BrainCancerFeaturizer,
                 regressor : BrainCancerRegressor,
                 n_domains : int,
                 hidden_size: int = 1024):
        super().__init__()
        self.featurizer = featurizer
        self.regressor  = regressor
        self.grl        = GradientReverseLayer()      # from your snippet
        self.grl.coeff = 1.0                  # initial coefficient for GRL
        # self.vec_pool   = nn.AdaptiveAvgPool3d(1)
        # self.scanner    =   ScannerBranch(out_ch=128)   # new scanner
        in_feat = self.featurizer.d_out                                     # ← discriminator input
        # Domain discriminator now expects a 256-D vector
        self.domain_classifier = DomainDiscriminator(
            in_feature = in_feat,
            n_domains  = n_domains,
            hidden_size= hidden_size,
            batch_norm = self.featurizer.batch_norm
        )
        self.n_domains = n_domains
        # self.domain_classifier = nn.Linear(in_feat, 15)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap      = self.featurizer(x)               # (B,256,D',H',W')
        y_pred    = self.regressor(fmap)             # (B,)
        # feat_vec  = self.vec_pool(fmap).flatten(1)   # (B,256)
        # ------------ domain path -------------------------------------------
        # scan_vec = self.scanner(x)            # (B,32)
        if self.grl.coeff <= 0.0:
            dom_pred = torch.empty(x.size(0), self.n_domains, device=x.device)
            return y_pred, dom_pred
        feat_vec = fmap.flatten(1)         # B×(256·7·9·7) = B×112 896
        feat_vec = self.grl(feat_vec)                # (B,256) – gradient-reversed
        dom_pred  = self.domain_classifier(feat_vec)  # (B,n_domains)
        return y_pred, dom_pred

    # ================================================================
    # 1)  Return *detached* parameters, grouped exactly like utils.py
    # ================================================================
    def get_params_for_layers(self):
        """
        Uses utils.get_layer_params_list / _dict so code written for the
        old BrainCancer model keeps working.

        Returns
        -------
        layers : List[List[Tensor]]
            Outer list follows layer insertion-order (conv1, conv2, …);
            inner list contains that layer's parameters in the order
            they appeared in `model.named_parameters()`.
        """
        # 1) Collect *all* parameters in creation order
        #    and wrap each tensor in its own single-element list
        self.layers = [[p] for _, p in self.named_parameters()]
        self.num_layers = len(self.layers)
        return self.layers


    # ================================================================
    # 2)  Copy an incoming list-of-lists back into live parameters
    # ================================================================
    @torch.no_grad()
    def receive_and_update_params(self, new_params_for_layers):
        """
        new_params_for_layers : same nested structure returned by
        `get_params_for_layers()` (List[List[Tensor | np.ndarray | scalar]])
        """

        # --------  flatten nested list into a single list of Tensors --------
        flat_new_params = []
        for layer_idx, layer_params in enumerate(new_params_for_layers):
            # Accept call-sites that might still send a flat list
            if not isinstance(layer_params, (list, tuple)):
                layer_params = [layer_params]

            for p_idx, p in enumerate(layer_params):
                if isinstance(p, torch.Tensor):
                    flat_new_params.append(p)
                elif isinstance(p, np.ndarray):
                    flat_new_params.append(torch.from_numpy(p) if p.shape else
                                        torch.tensor(p.item()))
                elif isinstance(p, (float, int, np.number)):
                    flat_new_params.append(torch.tensor(p))
                else:
                    raise TypeError(
                        f"Unexpected param type at layer {layer_idx} "
                        f"index {p_idx}: {type(p)}"
                    )

        # --------  sanity-check count  --------
        model_param_cnt = sum(1 for _ in self.parameters())
        if model_param_cnt != len(flat_new_params):
            raise ValueError(
                f"Param count mismatch: model has {model_param_cnt}, "
                f"received {len(flat_new_params)}"
            )

        # --------  copy the data into live parameters  --------
        for (name, param), new_p in zip(self.named_parameters(), flat_new_params):
            if param.shape != new_p.shape:
                raise ValueError(
                    f"[MISMATCH] {name}: expected {param.shape}, got {new_p.shape}"
                )
            param.copy_(new_p)          # in-place update

        # # (optional) keep a reference for debugging/inspection
        # self.layers = new_params_for_layers