import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def normalize_intensity(normalize_paras_):
    '''
    ToTensor(), Normalize()
    '''
    transform_list = [transforms.ToTensor(), transforms.Normalize(**__imagenet_stats)]
    return transforms.Compose(transform_list)


def to_tensor():
    return transforms.ToTensor()


def get_transform():
    '''
    API to get the transformation.
    return a list of transformations
    '''
    transform_list = normalize_intensity(__imagenet_stats)
    return transform_list


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


def save_image(folder, file_name, data, saver='matplotlib'):
    path_file = os.path.join(folder, file_name)
    if saver == 'matplotlib':
        if 'depth' in folder:
            cmap = matplotlib.cm.jet
            plt.imsave(path_file, data, cmap=cmap)
        else:
            plt.imsave(path_file, data)
    elif saver == 'opencv':
        cv2.imwrite(path_file, data)


"""
This implementation follows the idea from https://github.com/abdo-eldesokey/pncnn
However, the original version sorted all pixels of all confidence maps 
and then removed the fraction of lowest confidences gradually.
The computation is very expensive when the evaluating dataset is large. 
So, we compuate appoximately an estimation by considering batch-by-batch
"""
class SparsificationAverageMeter(object):
    def __init__(self, num_bins=100, top=1.0, uncert_type='c'):
        self.num_bins = num_bins
        self.ratio_removed = np.linspace(0, top, num_bins, endpoint=False)
        self.uncert_type = uncert_type
        self.reset()

    def reset(self):
        self.count = np.zeros([self.num_bins], np.uint64)
        self.rmse_err = np.zeros([self.num_bins], np.float32)
        self.rmse_err_by_cfd = np.zeros([self.num_bins], np.float32)

    def evaluate(self, depth, confidence, target):
        gt_depth, valid_mask = target
        depth = depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
        confidence = confidence[valid_mask]

        err_vec = (depth - gt_depth) ** 2

        err_vec_sorted, _ = torch.sort(err_vec)
        # print(' Done!')

        # Calculate the error when removing a fraction pixels with error
        n_valid_pixels = len(err_vec)
        rmse_err = []
        for i, r in enumerate(self.ratio_removed):
            mse_err_slice = err_vec_sorted[0:int((1 - r) * n_valid_pixels)]
            rmse_err.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

        rmse_err = np.array(rmse_err)  # / rmse_err[0]

        ###########################################

        # Sort by variance
        # print('Sorting Variance ...')
        if self.uncert_type == 'c':
            cfd_vec = torch.sqrt(confidence)
            _, cfd_vec_sorted_idxs = torch.sort(cfd_vec, descending=True)
        else:
            # var_vec = torch.exp(var_vec)
            cfd_vec = torch.sqrt(confidence)
            _, cfd_vec_sorted_idxs = torch.sort(cfd_vec, descending=False)
        # print(' Done!')

        # Sort error by confidence
        err_vec_sorted_by_cfd = err_vec[cfd_vec_sorted_idxs]

        rmse_err_by_cfd = []
        for i, r in enumerate(self.ratio_removed):
            mse_err_slice = err_vec_sorted_by_cfd[0:int((1 - r) * n_valid_pixels)]
            rmse_err_by_cfd.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

        rmse_err_by_cfd = np.array(rmse_err_by_cfd)

        self.rmse_err += rmse_err
        self.rmse_err_by_cfd += rmse_err_by_cfd
        self.count += valid_mask.size(0) # batch_size

    def result(self):
        self.rmse_err /= self.count
        self.rmse_err_by_cfd /= self.count

        # Normalize RMSE
        self.rmse_err /= self.rmse_err[0]
        self.rmse_err_by_cfd /= self.rmse_err_by_cfd[0]

        sparsification_error = self.rmse_err_by_cfd - self.rmse_err
        ause = np.trapz(sparsification_error, self.ratio_removed)
        return sparsification_error, ause
