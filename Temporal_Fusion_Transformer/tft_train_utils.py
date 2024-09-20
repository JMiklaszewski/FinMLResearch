import pickle
from typing import Dict,List,Tuple
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf,DictConfig
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import MSELoss

import sys
import os

sys.path.append(os.getcwd())

from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss

def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class DictDataSet(Dataset):
    def __init__(self, array_dict: Dict[str, np.ndarray]):
        self.keys_list = []
        for k, v in array_dict.items():
            self.keys_list.append(k)
            if np.issubdtype(v.dtype, np.dtype('bool')):
                setattr(self, k, torch.ByteTensor(v))
            elif np.issubdtype(v.dtype, np.int8):
                setattr(self, k, torch.CharTensor(v))
            elif np.issubdtype(v.dtype, np.int16):
                setattr(self, k, torch.ShortTensor(v))
            elif np.issubdtype(v.dtype, np.int32):
                setattr(self, k, torch.IntTensor(v))
            elif np.issubdtype(v.dtype, np.int64):
                setattr(self, k, torch.LongTensor(v))
            elif np.issubdtype(v.dtype, np.float32):
                setattr(self, k, torch.FloatTensor(v))
            elif np.issubdtype(v.dtype, np.float64):
                setattr(self, k, torch.DoubleTensor(v))
            else:
                setattr(self, k, torch.FloatTensor(v))

    def __getitem__(self, index):
        return {k: getattr(self, k)[index] for k in self.keys_list}

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]
    
def recycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_set_and_loaders(data_dict: Dict[str, np.ndarray],
                        shuffled_loader_config: Dict,
                        serial_loader_config: Dict,
                        ignore_keys: List[str] = None,
                        ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = DictDataSet({k:v for k,v in data_dict.items() if (ignore_keys and k not in ignore_keys)})
    loader = torch.utils.data.DataLoader(dataset,**shuffled_loader_config)
    serial_loader = torch.utils.data.DataLoader(dataset,**serial_loader_config)

    return dataset,iter(recycle(loader)),serial_loader

class QueueAggregator(object):
    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)

    def get(self):
        return self._queued_list

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

                        
def process_batch(batch: Dict[str,torch.tensor],
                  model: nn.Module,
                  quantiles_tensor: torch.tensor,
                  device:torch.device):
    if torch.cuda.is_available():
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch['target']

    predicted_quantiles = batch_outputs['predicted_quantiles']
    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_quantiles,
                                                              targets=labels,
                                                              desired_quantiles=quantiles_tensor)
    return q_loss, q_risk

def process_batch_class(batch: Dict[str, torch.tensor], 
                        model: nn.Module, 
                        device: torch.device,
                        loss = nn.CrossEntropyLoss()):

    loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch['target']

    predictions = batch_outputs['predicted_quantiles']

    return loss(predictions.squeeze(1), labels.squeeze(1)) # torch.squeeze - removes redundant middle dimention

from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from sklearn.metrics import classification_report
def cross_validate_model(dataset, model, num_epochs, num_classes, cv_split):
    # kfold = KFold(n_splits=num_splits, shuffle=False)
    
    fold_results = []
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_split):
        print(f'Fold {fold+1}/{len(list(cv_split))}')
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
        
        # Collect predictions and probabilities for CSV output
        model.eval()
        fold_predictions = []
        fold_probabilities = []
        fold_targets = []

        vars = ['static_feats_numeric', 'static_feats_categorical',
        'historical_ts_numeric', 'historical_ts_categorical',
        'future_ts_numeric', 'future_ts_categorical', 'target']
        
        for batch in val_loader:

            batch = {v:i for v,i in zip(vars, batch)}
            outputs = model(batch)
            
            logits = outputs['predicted_quantiles']  # Extract logits
            logits = torch.reshape(logits, (logits.shape[0], logits.shape[-1]))

            targets = batch['target'].flatten()

            if model.task_type == 'classification':
                fold_predictions.extend(torch.argmax(logits, dim=1).tolist())
            elif model.task_type == 'regression':
                fold_predictions.extend(logits.tolist())
            # fold_probabilities.extend(logits.tolist())  # Only take the last time step
            fold_targets.extend(targets.tolist())
        
        all_predictions.append(fold_predictions)
        # all_probabilities.append(fold_probabilities)
        all_targets.append(fold_targets)
    
    # Flatten lists for CSV creation
    all_predictions_flat = [item for sublist in all_predictions for item in sublist]
    # all_probabilities_flat = [item for sublist in all_probabilities for item in sublist]
    all_targets_flat = [item for sublist in all_targets for item in sublist]

    if model.task_type == 'classification':
        # Run classification report
        target_names = [f'Label_{i}' for i in range(num_classes)]
        fold_results = classification_report(all_targets_flat, all_predictions_flat, target_names=target_names)

    elif model.task_type == 'regression':
        fold_results = MSELoss()(torch.tensor(all_predictions_flat), torch.tensor(all_targets_flat))
    
    print('Cross-Validation results:')
    print(fold_results)

    return fold_results

' Monte Carlo Dropout - Prediction Uncertainties'
from tqdm import tqdm

def mc_dropout_predictions(model, data_loader, n_samples=100):
    
    vars = ['static_feats_numeric', 'static_feats_categorical',
        'historical_ts_numeric', 'historical_ts_categorical',
        'future_ts_numeric', 'future_ts_categorical', 'target']
    
    model.train()  # Ensure dropout is enabled
    predictions = []
    for _ in tqdm(range(n_samples)):
        batch_predictions = []
        for batch in data_loader:
            batch = {v:i for v,i in zip(vars, batch)}
            outputs = model(batch)
            logits = outputs['predicted_quantiles']  # Extract logits

            batch_predictions.append(logits.detach().cpu().numpy())
        predictions.append(np.concatenate(batch_predictions, axis=0))
    return np.array(predictions)

' Function to stack past values for n_past_values'
def stack_past_values(arr, n_past_values):

    n_samples, n_features = arr.shape

    if n_samples <= n_past_values: raise ValueError('Number of past values must be less than number of rows in array')
    
    out = np.zeros((n_samples - n_past_values, n_past_values, n_features))
    for i in range(n_past_values, n_samples):
        out[i - n_past_values] = arr[i - n_past_values:i]

    return out