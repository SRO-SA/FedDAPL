""" 3D brain age model"""
# mean absolute error
# adam optimizer
# learning rate 10^-4
# weight decay 10^-4
from torch import nn
import torch
from utils import get_layer_params_list, get_layer_params_dict, flatten_layer_param_list_for_model
from utils import log_print
import numpy as np
def conv_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_channel), nn.MaxPool3d(2, stride=2), nn.ReLU()
    )


class BrainCancer(nn.Module):
    def __init__(self, batch_size=1, initialization="default", device='cuda:0', **kwargs):
        super(BrainCancer, self).__init__()
        self.initialization = initialization
        self.d_out = 1
        self.batch_size = batch_size

        self.conv1 = conv_blk(1, 32) #1 ,32
        self.conv2 = conv_blk(32, 64)
        self.conv3 = conv_blk(64, 128)
        self.conv4 = conv_blk(128, 256)
        self.conv5 = conv_blk(256, 256)
        self.batch_norm=False

        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, stride=1),
            nn.InstanceNorm3d(64), nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 3, 2))
        )

        self.drop = nn.Identity()  # nn.Dropout3d(p=0.5)

        self.output = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        self.layers = []
        self.num_layers = 0
        self.param_struct = {}
        # self.init_weights()

    def init_weights(self):
        if self.initialization == "custom":
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out",
                        nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.output.bias, 62.68)

    def forward(self, x):
        # print("dsdsds: ", x.shape) # dsdsds:  torch.Size([32, 1, 1, 121, 145, 121])
        # dsdsds:  torch.Size([1, 1, 1, 121, 145, 121]) if batch_size == 1
        # print("y_pred in model: ", x)
        if torch.equal(torch.Tensor(list(x.shape)), torch.Tensor([1, 1, 1, 1, 121, 145, 121])):
            print("in if")
            x = torch.squeeze(x, dim = 0)
        x = x.view(-1, 1, 121, 145, 121)
        # print("in forward: ", x.shape)
        # print("y_pred in model: ", x)
        # torch.autograd.set_detect_anomaly(True)
        x = self.conv1(x)
        # print("in forward after conv1: ", x.shape)   
        # x = torch.squeeze(x)     
        # print("in forward after conv1 squeezed: ", x.shape)   
        # print("y_pred in model: ", x)
        x = self.conv2(x)
        # print("conv2 shape: ", x.shape)
        x = self.conv3(x)
        # print("conv3 shape: ", x.shape)
        x = self.conv4(x)
        # print("conv4 shape: ", x.shape)
        x = self.conv5(x)
        # print("conv5 shape: ", x.shape)
        x = self.conv6(x)
        # print("conv6 shape: ", x.shape)
        x = self.drop(x)
        # print("y_pred in model: ", x)
        # print("drop shape: ", x.shape)
        x = self.output(x)
        # print("y_pred in model: ", x)
        x = torch.squeeze(x, dim=[1,2,3])
        # print("y_pred in model: ", x)
        # print("final shape: ", x.shape)
        return x

    def get_params_for_layers(self):
        """
        Returns a list of parameter tensors for each layer in the model.
        The order of the list corresponds to the order of model.parameters().
        """
        # We clone/detach so that any in-place modifications won't
        # accidentally affect your model's gradient flow.
        # (and you won't end up returning references to the same leaf tensors.)
        self.layers = get_layer_params_list(self)
        self.param_struct = get_layer_params_dict(self)
        self.num_layers = len(self.layers)
        return self.layers
        
    @torch.no_grad()
    def receive_and_update_params(self, new_params_for_layers):
        """
        new_params_for_layers: List[List[Tensor]] with same structure as get_layer_params_list()
        """
        self.layers = new_params_for_layers  # Optional: store for debugging
        # for i, p in enumerate(new_params_for_layers):
        #     log_print(f"Param {i}: shape={p.shape}, type={type(p)}", context="RECIEVE AND UPDATE MODEL")
        # Flatten new params
        # log_print(f"Received {len(new_params_for_layers)} new params", context="MODEL")
        flat_new_params = []
        for i, p in enumerate(new_params_for_layers):
            if isinstance(p, torch.Tensor):
                flat_new_params.append(p)
            elif isinstance(p, np.ndarray):
                if p.shape == ():  # scalar numpy array
                    flat_new_params.append(torch.tensor(p.item()))  # turn into scalar tensor
                else:
                    flat_new_params.append(torch.from_numpy(p))
            elif isinstance(p, (float, int, np.number)):  # numpy.float32, float, int, etc.
                flat_new_params.append(torch.tensor(p))
            else:
                raise TypeError(f"Unexpected param type: {type(p)} at index {i}")
        # log_print(f"Received {len(flat_new_params)} flat params", context="MODEL")
        for i, (name, param) in enumerate(self.named_parameters()):
            new_param = flat_new_params[i]
            if param.shape != new_param.shape:
                print(f"[MISMATCH] {name}: expected {param.shape}, got {new_param.shape}")

        # Iterate over model's named parameters and copy new values
        for i, (name, param) in enumerate(self.named_parameters()):
            # log_print(f"Updating {name} with shape {param.shape}")
            new_param = flat_new_params[i]
            if param.shape != new_param.shape:
                raise ValueError(f"Shape mismatch for param {name}: expected {param.shape}, got {new_param.shape}")
            param.copy_(new_param)