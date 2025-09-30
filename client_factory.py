from doctest import debug
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import debug_function, log_print
import os
from client import FedClient, FlowerClientWrapper
from model import BrainCancer
from model_feature_regress import BrainCancerFeaturizer, BrainCancerRegressor, DANN3D

class ClientFactory:
    @debug_function(context="CLIENT FACTORY")
    def __init__(self, paths, server, clients_dataset, model_type, batch_size=1, epochs=10, client_cfg=None):
        """
        paths: list of paths for each client
        server: Server object
        clients_dataset: dictionary mapping client IDs to datasets
        batch_size: batch size for training
        epochs: number of epochs for training
        """
        self.epochs = epochs
        self.paths = paths
        self.clients_dataset = clients_dataset
        self.server = server
        self.batch_size = batch_size
        self.log_base_path = paths
        self.model_type = model_type
        self.model = None
        self.client_cfg = client_cfg

    @debug_function(context="CLIENT FACTORY")
    def __call__(self, cid: str):
        # log_print(f"[DEBUG] ClientFactory.__call__ called with cid={cid}")
        # log_print("The server Initial Paramter shape is: ", len(self.server.initial_dummy_paramters_list), context="SERVER")
        cid = int(cid)
        # log_print(f"[CLIENT FACTORY] ClientFactory.__call__ called with epoch={self.epochs}")
        log_path = self.paths[cid]
        fed_client = FedClient(
            client_id=cid,
            param_struct=self.server.initial_dummy_paramters_list,
            batch_size=self.batch_size,
            epochs = self.epochs,
            cfg=self.client_cfg
        )
        # log_print(f"[DEBUG] keys of clients_dataset: {list(self.clients_dataset.keys())}")
        if self.model_type=="DANN3D":
            # n_domains  = 15
            n_domains = 62
            feat_net   = BrainCancerFeaturizer(use_conv5=True)  # or False for conv4
            reg_head   = BrainCancerRegressor()
            model = DANN3D(feat_net, reg_head, n_domains, hidden_size=512)
            # if torch.cuda.is_available():
            #     model = model.cuda()
            self.model = model
            
        elif self.model_type=="Normal":
            model = BrainCancer()
            # if torch.cuda.is_available():
            #     model = model.cuda()
            self.model = model
            
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        fed_client.init_model(self.clients_dataset[str(cid)], self.model)
        # labels = self.clients_dataset[str(cid)]._metadata_array[:, 0].numpy()  # 0th col is 'site'
        # print("Site label range:", labels.min(), labels.max(),
        #     "| num_domains:", len(self.clients_dataset[str(cid)]._metadata_map['site']))
        # assert labels.max() < len(self.clients_dataset[str(cid)]._metadata_map['site'])
        # exit()
        return FlowerClientWrapper(fed_client, log_path=log_path).to_client()
    
    
class ValidationClientFactory:
    @debug_function(context="VALIDATION CLIENT FACTORY")
    def __init__(self, server, validation_dataset, model_type, batch_size=1, client_cfg=None):
        """
        paths: list of paths for each client
        server: Server object
        clients_dataset: dictionary mapping client IDs to datasets
        batch_size: batch size for training
        epochs: number of epochs for training
        """
        self.clients_dataset = validation_dataset
        self.server = server
        self.batch_size = batch_size
        self.model_type = model_type
        self.model = None
        self.client_cfg = client_cfg


    @debug_function(context="VALIDATION CLIENT FACTORY")
    def __call__(self, cid: str, validation_parameters=None):
        # log_print(f"[DEBUG] ClientFactory.__call__ called with cid={cid}")
        # log_print("The server Initial Paramter shape is: ", len(self.server.initial_dummy_paramters_list), context="SERVER")
        cid = int(cid)
        if self.model_type=="DANN3D":
            # n_domains  = 15
            n_domains = 62
            feat_net   = BrainCancerFeaturizer(use_conv5=True)  # or False for conv4
            reg_head   = BrainCancerRegressor()
            model = DANN3D(feat_net, reg_head, n_domains, hidden_size=512)
            # if torch.cuda.is_available():
            #     model = model.cuda()
            self.model = model
                
        elif self.model_type=="Normal":
            model = BrainCancer()
            # if torch.cuda.is_available():
            #     model = model.cuda()
            self.model = model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        fed_client = FedClient(
            client_id=cid,
            param_struct=validation_parameters, #Update!
            batch_size=self.batch_size,
            epochs=1,
            cfg=self.client_cfg
        )
        # Let the server use GPU if available
        if torch.cuda.is_available():
            fed_client.device_pref = "cuda"
        else:
            fed_client.device_pref = "cpu"        # log_print(f"[DEBUG] keys of clients_dataset: {list(self.clients_dataset.keys())}")
        fed_client.init_model(self.clients_dataset, model=self.model, is_train=False)
        return fed_client