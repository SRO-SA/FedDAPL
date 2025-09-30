# import os
# from pathlib import Path
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from wilds.datasets.wilds_dataset import WILDSDataset
# from wilds.common.metrics.all_metrics import MSE, PearsonCorrelation, MAE
# from wilds.common.grouper import CombinatorialGrouper
# from wilds.common.utils import subsample_idxs, shuffle_arr
# from collections import Counter

# DATASET = '2009-17'
# BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']


# DHS_SITES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
#             21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
#             31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
#             41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
#             51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
#             61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0]


# _SPLIT_DATA_60_TOTAL = {
#     'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
#             21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
#             31.0, 32.0, 33.0, 34.0, 36.0, 39.0, 40.0],
#     'val':[54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 61.0, 62.0, 63.0, 64.0],
#     'test':[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
#             51.0, 52.0, 53.0],
#     'val_id':[37.0, 38.0, 60.0, 65.0, 66.0, 67.0, 68.0, 69.0],
#     'test_id':[35.0]
# }

# _SPLIT_DATA_40_TOTAL = {
#     'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#             11.0, 12.0, 13.0, 14.0,],
#     'val':[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
#            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,],
#     'test':[37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 65.0],
#     'val_id':[34.0, 35.0, 36.0, 64.0],
#     'test_id':[46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
#                56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 66.0, 67.0, 
#                68.0]
# }

# _SPLIT_DATA_ORIGINAL_TOTAL = {
#     'train': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#               11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 20.0,
#               21.0, 22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 29.0,
#               30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
#               38.0, 40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
#               48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 56.0, 57.0,
#               59.0, 60.0, 61.0, 62.0, 63.0, 65.0, 66.0, 70.0],
#     'val_id': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#                12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 20.0, 21.0,
#                22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 29.0, 30.0,
#                31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
#                40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
#                49.0, 51.0, 52.0, 53.0, 56.0, 57.0, 59.0, 60.0,
#                61.0, 62.0, 63.0, 65.0, 66.0, 70.0],
#     'val': [15.0, 19.0, 36.0, 41.0, 55.0, 64.0]
# }

# _SPLIT_DATA_60 = {
#     'train': _SPLIT_DATA_60_TOTAL['train']+_SPLIT_DATA_60_TOTAL['val_id']+_SPLIT_DATA_60_TOTAL['test_id'],
#     'val':_SPLIT_DATA_60_TOTAL['val'],
#     'test':_SPLIT_DATA_60_TOTAL['test'],
# }

# _SPLIT_DATA_40 = {
#     'train': _SPLIT_DATA_40_TOTAL['train'],# + _SPLIT_DATA_40_TOTAL['val_id'] +_SPLIT_DATA_40_TOTAL['test_id'],
#     'val':_SPLIT_DATA_40_TOTAL['val'],# +_SPLIT_DATA_40_TOTAL['val_id'],
#     'test':_SPLIT_DATA_40_TOTAL['test'], # +_SPLIT_DATA_40_TOTAL['test_id'],
# }

# _SPLIT_DATA = _SPLIT_DATA_40

# # _SPLIT_DATA = {
# #     'train': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
# #             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
# #             21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
# #             31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
# #             41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,],
# #     #,"51.0", "52.0", "53.0", "54.0", "55.0"],
# #     'val': [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
# #             61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,],
# #     'test': [69.0, 70.0] #64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0
# # }

# # means and standard deviations calculated over the entire dataset (train + val + test),
# # with negative values set to 0, and ignoring any pixel that is 0 across all bands
# # all images have already been mean subtracted and normalized (x - mean) / std

# _MEANS_2009_17 = {
#     'BLUE':  0.059183,
#     'GREEN': 0.088619,
#     'RED':   0.104145,
#     'SWIR1': 0.246874,
#     'SWIR2': 0.168728,
#     'TEMP1': 299.078023,
#     'NIR':   0.253074,
#     'DMSP':  4.005496,
#     'VIIRS': 1.096089,
#     # 'NIGHTLIGHTS': 5.101585, # nightlights overall
# }

# _STD_DEVS_2009_17 = {
#     'BLUE':  0.022926,
#     'GREEN': 0.031880,
#     'RED':   0.051458,
#     'SWIR1': 0.088857,
#     'SWIR2': 0.083240,
#     'TEMP1': 4.300303,
#     'NIR':   0.058973,
#     'DMSP':  23.038301,
#     'VIIRS': 4.786354,
#     # 'NIGHTLIGHTS': 23.342916, # nightlights overall
# }


# # def split_by_countries(idxs, ood_countries, metadata):
# #     countries = np.asarray(metadata['country'].iloc[idxs])
# #     is_ood = np.any([(countries == country) for country in ood_countries], axis=0)
# #     return idxs[~is_ood], idxs[is_ood]

# def split_by_site(idxs, ood_sites, metadata):
#     # print("idxs: ", idxs)
#     # print("ood sites: ",ood_sites)
#     sites = np.asarray(metadata['site'].iloc[idxs])
#     # print("sites: ", sites)
#     is_ood = np.any([(sites == site) for site in ood_sites], axis=0)
#     return idxs[~is_ood], idxs[is_ood]

# class OpenBHBDataset(WILDSDataset):
#     """
#     The PovertyMap poverty measure prediction dataset.
#     This is a processed version of LandSat 5/7/8 satellite imagery originally from Google Earth Engine under the names `LANDSAT/LC08/C01/T1_SR`,`LANDSAT/LE07/C01/T1_SR`,`LANDSAT/LT05/C01/T1_SR`,
#     nighttime light imagery from the DMSP and VIIRS satellites (Google Earth Engine names `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4` and `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`)
#     and processed DHS survey metadata obtained from https://github.com/sustainlab-group/africa_poverty and originally from `https://dhsprogram.com/data/available-datasets.cfm`.

#     Supported `split_scheme`:
#         - 'official' and `countries`, which are equivalent
#         - 'mixed-to-test'

#     Input (x):
#         224 x 224 x 8 satellite image, with 7 channels from LandSat and 1 nighttime light channel from DMSP/VIIRS. Already mean/std normalized.

#     Output (y):
#         y is a real-valued asset wealth index. Higher index corresponds to more asset wealth.

#     Metadata:
#         each image is annotated with location coordinates (noised for anonymity), survey year, urban/rural classification, country, nighttime light mean, nighttime light median.

#     Website: https://github.com/sustainlab-group/africa_poverty

#     Original publication:
#     @article{yeh2020using,
#         author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
#         day = {22},
#         doi = {10.1038/s41467-020-16185-w},
#         issn = {2041-1723},
#         journal = {Nature Communications},
#         month = {5},
#         number = {1},
#         title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
#         url = {https://www.nature.com/articles/s41467-020-16185-w},
#         volume = {11},
#         year = {2020}
#     }

#     License:
#         LandSat/DMSP/VIIRS data is U.S. Public Domain.

#     """
#     _dataset_name = 'openBHB'
#     _versions_dict = {
#         '1.1': {
#             'download_url': 'https://worksheets.codalab.org/rest/bundles/0xfc0aa86ad9af4eb08c42dfc40eacf094/contents/blob/',
#             'compressed_size': 13_091_823_616}}

#     def __init__(self, version=None, root_dir='/rhome/ssafa013/bigdata/data', download=False,  # 'data'
#                  split_scheme='official',
#                  use_ood_val=True):
#         self._version = version
#         self._data_dir = self.initialize_data_dir(root_dir, download, harmonized=None) #_harmonized
#         # self._original_resolution = (224, 224)
#         print(self._data_dir)
#         self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
#         self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

#         if split_scheme == 'official':
#             split_scheme = 'sites'

#         if split_scheme == 'mixed-to-test':
#             self.oracle_training_set = True
#         elif split_scheme in ['official', 'sites']:
#             self.oracle_training_set = False
#         else:
#             raise ValueError("Split scheme not recognized")
#         self._split_scheme = split_scheme

#         fold = 'A'
#         self.root = Path(self._data_dir)
#         self.metadata = pd.read_csv(self.root / 'images/metadata.tsv', sep='\t')
#         # country folds, split off OOD
#         # country_folds = SURVEY_NAMES[f'2009-17{fold}']
#         # print('site max:', self.metadata['site'].max(), 'min: ', self.metadata['site'].min())
#         # print('study max:', self.metadata['study'].max(), 'min: ', self.metadata['study'].min())
#         self.metadata['site'] = self.metadata['site'] -1
#         self.metadata['study'] = self.metadata['study'] -1

#         site_folds = _SPLIT_DATA
#         # print("total images is: ", len(self.metadata))
#         self._split_array = -1 * np.ones(len(self.metadata))
        
        
#         insite_folds_split = np.arange(len(self.metadata))
#         # take the test countries to be ood
#         idxs_id, idxs_ood_test = split_by_site(insite_folds_split, site_folds['test'], self.metadata)
#         # print("ood test: ", idxs_ood_test)
#         # also create a validation OOD set
#         idxs_id, idxs_ood_val = split_by_site(idxs_id, site_folds['val'], self.metadata)
#         garbage, idxs_id = split_by_site(idxs_id, site_folds['train'], self.metadata)
#         # print("ood val: ", idxs_ood_val, "len val: ", len(idxs_ood_val)) # 387
#         for split in ['test', 'val', 'id_test', 'id_val', 'train']:
#             # keep ood for test, otherwise throw away ood data
#             if split == 'test':
#                 idxs = idxs_ood_test
#             elif split == 'val':
#                 idxs = idxs_ood_val
#             else:
#                 idxs = idxs_id
#                 num_eval_40 = 1196
#                 num_eval_60 = 797
#                 num_eval = 2 #num_eval_40
#                 # if oracle, sample from all countries
#                 if split == 'train' and self.oracle_training_set:
#                     idxs = subsample_idxs(insite_folds_split, num=len(idxs_id), seed=ord(fold))[num_eval:]
#                 elif split == 'train':
#                     idxs = subsample_idxs(idxs, take_rest=True, num=num_eval, seed=ord(fold))
#                 else:
#                     eval_idxs  = subsample_idxs(idxs, take_rest=False, num=num_eval, seed=ord(fold))

#                 if split != 'train':
#                     if split == 'id_val':
#                         idxs = eval_idxs[:num_eval//2]
#                     else:
#                         idxs = eval_idxs[num_eval//2:]
#             self._split_array[idxs] = self._split_dict[split]

#         unique, counts = np.unique(self._split_array, return_counts=True)
#         # print(dict(zip(unique, counts)))
#         if not use_ood_val:
#             self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
#             self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

#         self._y_array = torch.from_numpy(np.asarray(self.metadata['age'])[:, np.newaxis]).float()
#         self._y_size = 1
#         # add site group field
#         site_to_idx = {site: i for i, site in enumerate(DHS_SITES)}
        
#         # temp_metadata = self.metadata.copy()
#         # new_data = {'site':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#         #     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
#         #     21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
#         #     31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
#         #     41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
#         #     51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
#         #     61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0]}
#         # new_df = pd.DataFrame(new_data)
#         # mask = ~new_df['site'].isin(temp_metadata['site'])
#         # new_sites_to_add = new_df[mask]
#         # result_df = temp_metadata.append(new_sites_to_add, ignore_index=True)
#         # each_group_count = result_df.groupby('site', dropna=False, observed=False)['participant_id'].count().fillna(0).reset_index(name='count')
#         # pd.set_option('display.max_rows', 999)
#         # pd.set_option('display.max_columns', 999)
#         # pd.set_option('display.width', 999)
#         # print(each_group_count)
#         # df = pd.DataFrame(each_group_count)
#         # df.set_index('site', inplace=True)
#         # csv_filename = 'site_metadata_count.csv'
#         # df.to_csv(csv_filename)
        
        
#         self.metadata['site'] = [site_to_idx[site] for site in self.metadata['site'].tolist()]
#         self._metadata_map = {'site': DHS_SITES}
#         # self._metadata_array = torch.from_numpy(self.metadata[['site', 'age', 'study', 'participant_id']].astype(float).to_numpy())
#         self._metadata_array = torch.from_numpy(
#             self.metadata.assign(sex=self.metadata['sex'].map({'male': 1, 'female': 2}))
#                         [['site', 'age', 'study', 'participant_id', 'sex']]
#                         .astype(float)
#                         .to_numpy()
#         )
#         # rename wealthpooled to y
#         self._metadata_fields = ['site', 'y', 'study', 'participant_id', 'sex']

#         self._eval_grouper = CombinatorialGrouper(
#             dataset=self,
#             groupby_fields=['study'])

#         super().__init__(root_dir, download, split_scheme)

#     def get_input(self, idx):
#         """
#         Returns x for a given idx.
#         """
#         #print(idx)
#         participant_id = self.metadata['participant_id'][idx]
#         # print("participant_id", participant_id)
#         # _preproc-quasiraw_T1w
#         # _preproc-cat12vbm_desc-gm_T1w
#         # img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
#         try:
#             img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
#         except FileNotFoundError:
#             print("File not found!, using different root directory  ", participant_id)
#             new_dir = os.path.join('/rhome/ssafa013/bigdata/data', f'{self.dataset_name}_v{self.version}')
#             img = np.load(Path(new_dir) / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
#             pass
#         img = torch.from_numpy(img).float()

#         return img

#     def eval(self, y_pred, y_true, metadata, prediction_fn=None):
#         """
#         Computes all evaluation metrics.
#         Args:
#             - y_pred (Tensor): Predictions from a model
#             - y_true (LongTensor): Ground-truth values
#             - metadata (Tensor): Metadata
#             - prediction_fn (function): Only None supported
#         Output:
#             - results (dictionary): Dictionary of evaluation metrics
#             - results_str (str): String summarizing the evaluation metrics
#         """
#         assert prediction_fn is None, "PovertyMapDataset.eval() does not support prediction_fn"

#         metrics = [MSE(), MAE()]

#         all_results = {}
#         all_results_str = ''
#         for metric in metrics:
#             results, results_str = self.standard_group_eval(
#                 metric,
#                 self._eval_grouper,
#                 y_pred, y_true, metadata)
#             all_results.update(results)
#             all_results_str += results_str
#         return all_results, all_results_str



import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.metrics.all_metrics import MSE, PearsonCorrelation, MAE
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import subsample_idxs, shuffle_arr
from collections import Counter
from typing import Optional


DATASET = '2009-17'
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']


DHS_SITES = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
    41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
    51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
    61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0
]

# ------------------- existing split sets you already had -------------------
_SPLIT_DATA_60_TOTAL = {
    'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 36.0, 39.0, 40.0],
    'val':[54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 61.0, 62.0, 63.0, 64.0],
    'test':[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
            51.0, 52.0, 53.0],
    'val_id':[37.0, 38.0, 60.0, 65.0, 66.0, 67.0, 68.0, 69.0],
    'test_id':[35.0]
}

_SPLIT_DATA_40_TOTAL = {
    'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0,],
    'val':[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
           25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,],
    'test':[37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 65.0],
    'val_id':[34.0, 35.0, 36.0, 64.0],
    'test_id':[46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
               56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 66.0, 67.0, 
               68.0]
}

# NEW: Your "official OpenBHB" split mapping based on participants_train/val.tsv
_SPLIT_DATA_ORIGINAL_TOTAL = {
    # train-val-id_val cover the in-distribution sites (as in your note)
    'train': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
              11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 20.0,
              21.0, 22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 29.0,
              30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
              38.0, 40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
              48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 56.0, 57.0,
              59.0, 60.0, 61.0, 62.0, 63.0, 65.0, 66.0, 70.0],
    'val_id': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 20.0, 21.0,
               22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 29.0, 30.0,
               31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
               40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
               49.0, 51.0, 52.0, 53.0, 56.0, 57.0, 59.0, 60.0,
               61.0, 62.0, 63.0, 65.0, 66.0, 70.0],
    # OOD sites for external validation (from your earlier site list)
    'val': [15.0, 19.0, 36.0, 41.0, 55.0, 64.0]
}

_SPLIT_DATA_60 = {
    'train': _SPLIT_DATA_60_TOTAL['train']+_SPLIT_DATA_60_TOTAL['val_id']+_SPLIT_DATA_60_TOTAL['test_id'],
    'val':_SPLIT_DATA_60_TOTAL['val'],
    'test':_SPLIT_DATA_60_TOTAL['test'],
}

_SPLIT_DATA_40 = {
    'train': _SPLIT_DATA_40_TOTAL['train'],
    'val':_SPLIT_DATA_40_TOTAL['val'],
    'test':_SPLIT_DATA_40_TOTAL['test'],
}

# Default unchanged
# _SPLIT_DATA = _SPLIT_DATA_40
_SPLIT_DATA = _SPLIT_DATA_ORIGINAL_TOTAL

def split_by_site(idxs, ood_sites, metadata):
    sites = np.asarray(metadata['site'].iloc[idxs])
    is_ood = np.any([(sites == site) for site in ood_sites], axis=0)
    return idxs[~is_ood], idxs[is_ood]


class OpenBHBDataset(WILDSDataset):
    _dataset_name = 'openBHB'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xfc0aa86ad9af4eb08c42dfc40eacf094/contents/blob/',
            'compressed_size': 13_091_823_616}
    }

    def __init__(
        self,
        version=None,
        root_dir='/rhome/ssafa013/bigdata/data',
        download=False,
        split_scheme='official',
        use_ood_val=True,
        # NEW: toggle to use participants_train/val.tsv instead of images/metadata.tsv + custom site splits
        use_participants_tsv: bool = False,
        participants_train_path: Optional[Path] = None,
        participants_val_path: Optional[Path] = None,
    ):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download, harmonized=None)  # unchanged
        print(self._data_dir)
        
        use_participants_tsv = True
        # Keep original split dictionary & names (so downstream code doesn't break)
        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

        if split_scheme == 'official':
            split_scheme = 'sites'

        if split_scheme == 'mixed-to-test':
            self.oracle_training_set = True
        elif split_scheme in ['official', 'sites']:
            self.oracle_training_set = False
        else:
            raise ValueError('Split scheme not recognized')
        self._split_scheme = split_scheme

        fold = 'A'
        self.root = Path(self._data_dir)

        # ---------------- NEW BRANCH: use participants_train/val.tsv ----------------
        if use_participants_tsv:
            # Resolve paths or use defaults under root
            train_tsv = participants_train_path or (self.root / 'participants_train.tsv')
            val_tsv = participants_val_path or (self.root / 'participants_val.tsv')

            train_df = pd.read_csv(train_tsv, sep='\t').copy()
            val_df = pd.read_csv(val_tsv, sep='\t').copy()

            # Mark splits according to official OpenBHB: train, internal val, external val
            train_df['split_source'] = 'train'
            if 'split' not in val_df.columns:
                raise ValueError("participants_val.tsv must have a 'split' column with values 'internal_test'/'external_test'.")
            # Map to our internal split names (keeping WILDS-style indices)
            map_val = {
                'internal_test': 'id_val',   # In-distribution validation (ID Val)
                'external_test': 'val',      # OOD validation (external) -> use 'val' (no 'test' split here)
            }
            val_df['split_source'] = val_df['split'].map(map_val)

            # Align with the original metadata expectations
            combined = pd.concat([train_df, val_df], ignore_index=True)

            # Ensure zero-based site & study like your original code
            combined['site'] = combined['site'] - 1
            combined['study'] = combined['study'] - 1

            self.metadata = combined

            # Build split array directly from split_source
            self._split_array = -1 * np.ones(len(self.metadata))
            for split_name, split_idx in self._split_dict.items():
                mask = self.metadata.get('split_source', '') == split_name
                if mask is not None:
                    self._split_array[mask.to_numpy()] = split_idx

            # If you also want to keep an optional ID test (empty here), it remains -1/no rows.

        # ---------------- ORIGINAL BRANCH: keep current functionality ----------------
        else:
            # Original metadata location & processing
            self.metadata = pd.read_csv(self.root / 'images/metadata.tsv', sep='\t')
            self.metadata['site'] = self.metadata['site'] - 1
            self.metadata['study'] = self.metadata['study'] - 1

            site_folds = _SPLIT_DATA
            self._split_array = -1 * np.ones(len(self.metadata))

            insite_folds_split = np.arange(len(self.metadata))
            idxs_id, idxs_ood_test = split_by_site(insite_folds_split, site_folds['test'], self.metadata)
            idxs_id, idxs_ood_val = split_by_site(idxs_id, site_folds['val'], self.metadata)
            _, idxs_id = split_by_site(idxs_id, site_folds['train'], self.metadata)

            for split in ['test', 'val', 'id_test', 'id_val', 'train']:
                if split == 'test':
                    idxs = idxs_ood_test
                elif split == 'val':
                    idxs = idxs_ood_val
                else:
                    idxs = idxs_id
                    num_eval = 2  # unchanged default from your code
                    if split == 'train' and self.oracle_training_set:
                        idxs = subsample_idxs(insite_folds_split, num=len(idxs_id), seed=ord(fold))[num_eval:]
                    elif split == 'train':
                        idxs = subsample_idxs(idxs, take_rest=True, num=num_eval, seed=ord(fold))
                    else:
                        eval_idxs = subsample_idxs(idxs, take_rest=False, num=num_eval, seed=ord(fold))
                    if split != 'train':
                        if split == 'id_val':
                            idxs = eval_idxs[: num_eval // 2]
                        else:
                            idxs = eval_idxs[num_eval // 2 :]
                self._split_array[idxs] = self._split_dict[split]

            if not use_ood_val:
                self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
                self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        # ---------------- Common: labels, metadata map, arrays ----------------
        

        self._y_array = torch.from_numpy(np.asarray(self.metadata['age'])[:, np.newaxis]).float()
        self._y_size = 1

        # site reindexing to contiguous [0..len(DHS_SITES)-1]
        # site_to_idx = {site: i for i, site in enumerate(DHS_SITES)}
        # self._metadata_map = {'site': DHS_SITES}
        if use_participants_tsv:
            present_sites = sorted(self.metadata['site'].unique().tolist())
            site_to_idx = {s: i for i, s in enumerate(present_sites)}
            self._metadata_map = {'site': present_sites}   # length = K
        else:
            site_to_idx = {site: i for i, site in enumerate(DHS_SITES)}
            self._metadata_map = {'site': DHS_SITES}
            
        self.metadata['site'] = [site_to_idx[site] for site in self.metadata['site'].tolist()]

        # numeric sex: {'male':1, 'female':2}
        self._metadata_array = torch.from_numpy(
            self.metadata.assign(sex=self.metadata['sex'].map({'male': 1, 'female': 2}))
                        [['site', 'age', 'study', 'participant_id', 'sex']]
                        .astype(float)
                        .to_numpy()
        )
        labels = self._metadata_array[:, 0].numpy()  # 0th col is 'site'
        print("Site label range:", labels.min(), labels.max(),
            "| num_domains:", len(self._metadata_map['site']))
        assert labels.max() < len(self._metadata_map['site'])
        self._metadata_fields = ['site', 'y', 'study', 'participant_id', 'sex']

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=['study'])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """Returns x (image) for a given idx."""
        participant_id = self.metadata['participant_id'][idx]
        try:
            img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
        except FileNotFoundError:
            print('File not found!, using different root directory ', participant_id)
            new_dir = os.path.join('/rhome/ssafa013/bigdata/data', f'{self.dataset_name}_v{self.version}')
            img = np.load(Path(new_dir) / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
        return torch.from_numpy(img).float()

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Computes evaluation metrics (MSE, MAE) with standard_group_eval."""
        assert prediction_fn is None, 'OpenBHBDataset.eval() does not support prediction_fn'
        metrics = [MSE(), MAE()]
        all_results, all_results_str = {}, ''
        for metric in metrics:
            results, results_str = self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)
            all_results.update(results)
            all_results_str += results_str
        return all_results, all_results_str
