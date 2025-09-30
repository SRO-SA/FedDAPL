#!/usr/bin/env python
"""
federated_splits.py

Class-based approach to creating federated splits for the OpenBHBDataset.
"""

import torch
import numpy as np
from pathlib import Path
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset
from utils import log_print

# Adjust this import to match where your OpenBHBDataset code is located.
from DataLoader import OpenBHBDataset

# Default site-to-client grouping:
# _SPLIT_DATA_BY_CLIENT_5 = { 
#     '0': [0.0, 1.0, 2.0,],
#     '1': [ 3.0, 4.0, 5.0],
#     '2': [ 6.0, 7.0, 8.0],
#     '3': [ 9.0, 10.0,11.0],
#     '4': [ 12.0, 13.0, 14.0,]
# }

_SPLIT_DATA_BY_CLIENT_5 = { 
    '0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    '1': [ 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    '2': [ 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
    '3': [ 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0],
    '4': [ 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0]
}


class FederatedOpenBHBSplitter:
    """
    Class to load the OpenBHBDataset and split the 'train' subset
    among multiple clients according to site ID.
    """

    def __init__(
        self,
        root_dir: str = '/rhome/ssafa013/bigdata/data',
        split_scheme: str = 'official',
        use_ood_val: bool = True
    ):
        """
        Initializes the dataset from OpenBHBDataset and sets up the
        standard train/val/test splits.
        """
        self.root_dir = root_dir
        self.split_scheme = split_scheme
        self.use_ood_val = use_ood_val
        
        # 1) Instantiate your dataset
        self.dataset = OpenBHBDataset(
            root_dir=self.root_dir,
            split_scheme=self.split_scheme,
            use_ood_val=self.use_ood_val
        )
        log_print("Loaded dataset with total length:", len(self.dataset))

        # 2) Grab the standard WILDS subsets
        self.train_dataset = self.dataset.get_subset('train')
        self.val_dataset   = self.dataset.get_subset('val')
        self.test_dataset  = self.dataset.get_subset('test')

        log_print("Train subset size:", len(self.train_dataset), context="DATASET SPLIT")
        log_print("Val subset size:  ", len(self.val_dataset), context="DATASET SPLIT")
        log_print("Test subset size: ", len(self.test_dataset), context="DATASET SPLIT")

    def get_federated_splits(
        self,
        client_site_dict: dict = None
    ):
        """
        Returns:
            - client_train_datasets: dict of client_id -> WILDS subset for training
            - val_dataset: shared WILDS subset for validation
            - test_dataset: shared WILDS subset for testing
        """
        if client_site_dict is None:
            # Use the default dictionary if none is provided
            client_site_dict = _SPLIT_DATA_BY_CLIENT_5
        
        # 3) Create per-client training splits by site
        client_train_datasets = {}
        
        # The train dataset is a "subset" of the original dataset, so to
        # filter further we need the underlying metadata + the current indices.
        train_metadata = self.train_dataset.dataset.metadata
        train_indices  = self.train_dataset.indices  # Indices in the original dataset

        for client_id, site_list in client_site_dict.items():
            # We'll gather the indices that match any site in site_list
            client_indices = []
            for idx in train_indices:
                # "idx" is the original dataset index; check the site in metadata
                example_site = train_metadata.iloc[idx]['site']
                if example_site in site_list:
                    client_indices.append(idx)

            # Convert that list of indices into a WILDS subset
            # Instead of .get_subset(indices=...), build a WILDSSubset directly
            client_train_dataset = WILDSSubset(
                dataset=self.dataset,
                indices=client_indices,
                transform=self.train_dataset.transform
            )
            client_train_datasets[client_id] = client_train_dataset

            log_print(f"Client {client_id} train set size: {len(client_train_dataset)}", context="DATASET SPLIT")

        # Return the dictionary plus the shared val/test
        return client_train_datasets, self.val_dataset, self.test_dataset
