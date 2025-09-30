
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class IdentityScanner(nn.Module):
    """
    No learned convolutions.
    1.  Average-pool shrinks the volume by fixed factors
    2.  Flatten → vector
    """
    def __init__(self,
                 pool_kernel=(2, 2, 2),   # (D, H, W) reduction factors
                 ):
        super().__init__()

        self.pool = nn.AvgPool3d(kernel_size=pool_kernel,
                                 stride=pool_kernel)

        # original volume: 121×145×121  →  after pool: 30×29×30
        d, h, w  = [s // k for s, k in zip((121, 145, 121), pool_kernel)]
        flat_dim = d * h * w        # 26 100

        # self.classifier = nn.Sequential(
        #     nn.Linear(flat_dim, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, n_domains)
        # )

        self.d_out = flat_dim        # for bookkeeping

    def forward(self, x):            # x : (B,1,121,145,121)
        x = x.squeeze(1)          # drop the *second* axis (keeps batch)
        x = self.pool(x)             # (B,1,30,29,30)
        x = x.flatten(1)             # (B,26 100)
        return x
    
class ScannerBranch(nn.Module):
    """
    Tiny 3-D CNN   +  masked global mean
    Only non-zero voxels contribute to the feature vector.
    """
class ScannerBranch(nn.Module):
    """
    Tiny 3-D CNN that *preserves* large-scale intensity patterns.
    No instance-norm, no per-image normalisation, no global average.
    Output is the flattened conv-3 feature map (spatial info retained).
    """
    def __init__(self, out_ch: int = 32):
        super().__init__()

        # much gentler down-sampling: only TWO stride-2 ops
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2)   # 1×121×145×121
        self.pool1 = nn.MaxPool3d(2)                                         # → 16×60×72×60

        self.conv2 = nn.Conv3d(16, 24, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)                                         # → 24×30×36×30

        self.conv3 = nn.Conv3d(24, out_ch, kernel_size=3, stride=1, padding=1)   # keep spatial

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                nn.init.zeros_(m.bias)

        # feature-vector length the discriminator will see
        self.d_out = out_ch * 30 * 36 * 30          #  =  777 600
                        # 128

    # -------- helper -------------------------------------------------------
    @staticmethod
    def _masked_mean(feat, mask, eps: float = 1e-6):
        """feat, mask have identical shape  (B,C,D,H,W)."""
        num = (feat * mask).sum(dim=(2, 3, 4))
        den = mask.sum(dim=(2, 3, 4)).clamp_min(eps)
        return num / den                              # (B,C)

    # -----------------------------------------------------------------------
    def forward(self, x):                           # x : (B,1,1,121,145,121)
        x = x.squeeze(1)          # drop the *second* axis (keeps batch)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))                   #  (B,out_ch,30,36,30)
        return x.flatten(1)                         #  (B,777 600)
    
class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 256, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )
            # super(DomainDiscriminator, self).__init__(
            #     nn.Linear(in_feature, hidden_size),
            #     nn.ReLU(inplace=True),
            #     nn.GroupNorm(32, hidden_size),  # GroupNorm instead of BatchNorm
            #     nn.Linear(hidden_size, hidden_size),
            #     nn.ReLU(inplace=True),
            #     nn.GroupNorm(32, hidden_size),  # GroupNorm instead of BatchNorm
            #     nn.Linear(hidden_size, n_domains),
            # )
            # super(DomainDiscriminator, self).__init__(
            #     nn.Linear(in_feature, hidden_size),
            #     nn.ReLU(inplace=True),
            #     nn.InstanceNorm1d(hidden_size),  # InstanceNorm1d instead of BatchNorm
            #     nn.Linear(hidden_size, hidden_size),
            #     nn.ReLU(inplace=True),
            #     nn.InstanceNorm1d(hidden_size),  # InstanceNorm1d instead of BatchNorm
            #     nn.Linear(hidden_size, n_domains),
            # )

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Forward pass of the domain discriminator.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (minibatch, in_feature).

    #     Returns:
    #         torch.Tensor: Output tensor of shape (minibatch, n_domains).
    #     """
    #     return super(DomainDiscriminator, self).forward(x)

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self, coeff=1.0):
        super(GradientReverseLayer, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.coeff)


# class DomainAdversarialNetwork(nn.Module):
#     def __init__(self, featurizer, classifier, n_domains):
#         super().__init__()
#         self.featurizer = featurizer
#         self.classifier = classifier
#         print("featurizer domain: ", featurizer.d_out, "number of domains: ", n_domains)
#         print("batch norm: ", featurizer.batch_norm)
#         self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains, batch_norm=featurizer.batch_norm)
#         self.gradient_reverse_layer = GradientReverseLayer()

#     def forward(self, input):
#         features = self.featurizer(input)
#         y_pred = self.classifier(features)
#         features = self.gradient_reverse_layer(features)
#         print(features.shape)
#         domains_pred = self.domain_classifier(features)
#         return y_pred, domains_pred

#     def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
#         """
#         Adapted from https://github.com/thuml/Transfer-Learning-Library

#         A parameter list which decides optimization hyper-parameters,
#         such as the relative learning rate of each layer
#         """
#         # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
#         # feature extractor for better accuracy by default. For our implementation, we allow the learning
#         # rates to be passed in separately for featurizer and classifier.
#         params = [
#             {"params": self.featurizer.parameters(), "lr": featurizer_lr},
#             {"params": self.classifier.parameters(), "lr": classifier_lr},
#         ]
#         return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)



# --- add at top of server.py -----------------------------------------------
class ParamDiscriminator(torch.nn.Module):
    """Takes a *flattened* parameter vector and predicts the domain ID."""
    def __init__(self, d_in, n_domains, hidden=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_domains)
        )

    def forward(self, θ_flat):
        return self.net(θ_flat)
# ---------------------------------------------------------------------------