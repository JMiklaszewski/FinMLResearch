MODEL_ALIAS = 'TFT_Class'
NUM_WORKERS = 1

#------------------------------------
# ENVIRONMENT SETUP
#------------------------------------

# Disable warnings / messages from lightning
import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(0)

import sys
import os
cur_dir = os.getcwd()
# Add the current directory to system path (including parent folder)
sys.path.append(cur_dir)
sys.path.append(os.getcwd())
sys.path.append(os.path.split(os.getcwd())[0])
sys.path.append(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0]) # master drive

import pandas as pd
import numpy as np

from tft_pl import TemporalFusionTransformer
# from pl_model_utils import TimeSeriesDataModule, TimeSeriesDataset
from tft_train_utils import cross_validate_model
from mc_dropout_tft import mc_dropout
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.functional import nll_loss
from pytorch_lightning import Trainer
# from mc_dropout import mc_dropout
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.metrics import log_loss
import optuna
from tqdm import tqdm
from torch import save
import h5py
from time import time, ctime
from omegaconf import OmegaConf

from CPCV.cpcv import CombinatorialPurgedGroupKFold
from sklearn.model_selection import TimeSeriesSplit
from tft_train_utils import stack_past_values

# PLEASE MAKE SURE WE USE SEED FOR REPRODUCABILITY
seed_everything(123, workers=True)

from lightning.pytorch.utilities import disable_possible_user_warnings
disable_possible_user_warnings()
#------------------------------------
# SAMPLE DATA - REPLACE WITH REAL DATA
#------------------------------------

def generate_random_data(n_rows, n_classes):
    # Prepare sample data
    timestamp = pd.date_range(start='2020-01-01', periods=n_rows, freq='D')
    time_series = pd.DataFrame({'values': np.random.randn(n_rows)}, index=timestamp)
    labels = pd.DataFrame({'label': np.random.randint(0, n_classes, size=n_rows)}, index=timestamp)
    ext_features = pd.DataFrame({
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows)
        }, index=timestamp)

    return time_series.join(labels).join(ext_features)

#------------------------------------
# HYPERPARAMETER TUNING WITH OPTUNA
#------------------------------------

def tft_objective(trial, configuration, train_data):

    vars = ['static_feats_numeric', 'static_feats_categorical',
        'historical_ts_numeric', 'historical_ts_categorical',
        'future_ts_numeric', 'future_ts_categorical', 'target']

    config = configuration
    dataset = train_data

    # Suggest hyperparameters
    lr = trial.suggest_categorical('lr', [1e-5, 1e-3, 1e-2])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    dropout_prob = trial.suggest_categorical('dropout_prob', [0.1, 0.3, 0.5])
    hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
    lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2, 4])
    # classifier_units = trial.suggest_categorical('classifier_units', [16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    config['optimization']['learning_rate'] = lr
    config['optimization']['batch_size'] = batch_size
    config['model']['dropout'] = dropout_prob
    config['model']['state_size'] = hidden_units
    config['model']['lstm_layers'] = lstm_layers
    config['model']['attention_heads'] = num_heads

    # Initialize the model with suggested hyperparameters
    model = TemporalFusionTransformer(config=OmegaConf.create(config))

    # Time series split
    kfold = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[EarlyStopping(monitor='train_loss', patience=5, mode='min')],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )

        # Train the model
        trainer.fit(model, train_loader)

        # Validate the model

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {v:i for v,i in zip(vars, batch)}
                logits = model(batch)['predicted_quantiles']
                # preds = torch.argmax(classification, dim=1)
                all_preds.extend(logits.squeeze(1).cpu().numpy())
                all_targets.extend(batch['target'].flatten().cpu().numpy())

        # val_predictions = trainer.predict(model, val_loader)
        # val_predictions = torch.cat([x for x in val_predictions], dim=0).numpy()
        
        val_loss = nll_loss(torch.tensor(all_preds), torch.tensor(all_targets))
        cv_scores.append(val_loss)

    return np.mean(cv_scores)

def tft_classification(model_id):
    model_label = MODEL_ALIAS + '_' + model_id
    print('=' * 50)
    print(f'{ctime()} | [{model_label}] Running...')
    print('=' * 50)

    N_ROWS = 1500
    N_CLASSES = 4

    # Specify input - make sure to use real data
    df = generate_random_data(N_ROWS, N_CLASSES)

    train_data = df[:500]
    val_data = df[500:1000]
    test_data = df[1000:]

    # Data properties for TFT-type architecture
    data_props = {'num_historical_numeric': 2,
                'num_historical_categorical': 6,
                'num_static_numeric': 10,
                'num_static_categorical': 11,
                'num_future_numeric': 2,
                'num_future_categorical': 3,
                'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(), # cardinalities - ie. how many categories each variable has 
                'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                'num_classes': N_CLASSES
                }
    
    # create batch
    batch_size = 32
    historical_steps = 90
    future_steps = 1

    # read features and target label
    feat_train = test_data.loc[:,['feature1', 'feature2']]
    label_train = test_data['label'][-(len(train_data)-historical_steps):]

    feat_val = val_data.loc[:,['feature1', 'feature2']]
    label_val = val_data['label'][-(len(val_data)-historical_steps):]

    feat_test = test_data.loc[:,['feature1', 'feature2']]
    label_test = test_data['label'][-(len(test_data)-historical_steps):]

    n_rows = N_ROWS
    n_obs_train = len(train_data) - historical_steps
    n_obs_val = len(val_data) - historical_steps
    n_obs_test = len(test_data) - historical_steps

    batch_train = {
            'static_feats_numeric': torch.rand(n_obs_train, data_props['num_static_numeric'],dtype=torch.float32),
            'static_feats_categorical': torch.stack([torch.randint(c, size=(n_obs_train,)) for c in data_props['static_categorical_cardinalities']], dim=-1).type(torch.LongTensor),

            'historical_ts_numeric': torch.tensor(stack_past_values(feat_train.values, historical_steps), dtype=torch.float32),
            'historical_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_train, historical_steps)) for c in data_props['historical_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'future_ts_numeric': torch.rand(n_obs_train, future_steps, data_props['num_future_numeric'],dtype=torch.float32),
            'future_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_train, future_steps)) for c in data_props['future_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'target' : torch.reshape(torch.tensor(label_train[-(n_obs_train):].values, dtype=torch.int64), (n_obs_train, future_steps))
        }

    batch_test = {
            'static_feats_numeric': torch.rand(n_obs_test, data_props['num_static_numeric'], dtype=torch.float32),
            'static_feats_categorical': torch.stack([torch.randint(c, size=(n_obs_test,)) for c in data_props['static_categorical_cardinalities']], dim=-1).type(torch.LongTensor),

            'historical_ts_numeric': torch.tensor(stack_past_values(feat_test.values, historical_steps), dtype=torch.float32),
            'historical_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_test, historical_steps)) for c in data_props['historical_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'future_ts_numeric': torch.rand(n_obs_test, future_steps, data_props['num_future_numeric'], dtype=torch.float32),
            'future_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_test, future_steps)) for c in data_props['future_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'target' : torch.reshape(torch.tensor(label_test[-(n_obs_test):].values, dtype=torch.int64), (n_obs_test, future_steps))
        }

    batch_val = {
            'static_feats_numeric': torch.rand(n_obs_val, data_props['num_static_numeric'],dtype=torch.float32),
            'static_feats_categorical': torch.stack([torch.randint(c, size=(n_obs_val,)) for c in data_props['static_categorical_cardinalities']], dim=-1).type(torch.LongTensor),

            'historical_ts_numeric': torch.tensor(stack_past_values(feat_val.values, historical_steps), dtype=torch.float32),
            'historical_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_val, historical_steps)) for c in data_props['historical_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'future_ts_numeric': torch.rand(n_obs_val, future_steps, data_props['num_future_numeric'], dtype=torch.float32),
            'future_ts_categorical': torch.stack([torch.randint(c, size=(n_obs_val, future_steps)) for c in data_props['future_categorical_cardinalities']], dim=-1).type(torch.LongTensor),
            
            'target' : torch.reshape(torch.tensor(label_val[-(n_obs_val):].values, dtype=torch.int64), (n_obs_val, future_steps))
        }
    
    configuration = {
            'model':
                {
                    'dropout': 0.05,
                    'state_size': 64,
                    # 'output_quantiles': [0.1, 0.5, 0.9],
                    'lstm_layers': 2,
                    'attention_heads': 4
                },
            'optimization':
            {
                'batch_size': batch_size,
                'learning_rate': 1e-3,
                'max_grad_norm': 1.0
            },
            # these arguments are related to possible extensions of the model class
            'task_type': 'classification',
            'target_window_start': None,
            'data_props': data_props
        }
    
    train_data = TensorDataset(
        batch_train['static_feats_numeric'], batch_train['static_feats_categorical'],
        batch_train['historical_ts_numeric'], batch_train['historical_ts_categorical'],
        batch_train['future_ts_numeric'], batch_train['future_ts_categorical'],
        batch_train['target']
        )

    val_data = TensorDataset(
        batch_val['static_feats_numeric'], batch_val['static_feats_categorical'],
        batch_val['historical_ts_numeric'], batch_val['historical_ts_categorical'],
        batch_val['future_ts_numeric'], batch_val['future_ts_categorical'],
        batch_val['target']
        )

    test_data = TensorDataset(
        batch_test['static_feats_numeric'], batch_test['static_feats_categorical'],
        batch_test['historical_ts_numeric'], batch_test['historical_ts_categorical'],
        batch_test['future_ts_numeric'], batch_test['future_ts_categorical'],
        batch_test['target']
        )

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    print(f'{ctime()} | [{model_label}] Optimizing parameters...')
    # Run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: tft_objective(trial, configuration, train_data), n_trials=1)

    # Print best hyperparameters
    print(f"{ctime()} | [{model_label}] Best hyperparameters:", study.best_params)

    # Speciy the Autoencoder Model with optmial hyperparameters
    configuration['optimization']['learning_rate'] = study.best_params['lr']
    configuration['optimization']['batch_size'] = study.best_params['batch_size']
    configuration['model']['dropout'] = study.best_params['dropout_prob']
    configuration['model']['state_size'] = study.best_params['hidden_units']
    configuration['model']['lstm_layers'] = study.best_params['lstm_layers']
    configuration['model']['attention_heads'] = study.best_params['num_heads']

    tft_model = TemporalFusionTransformer(config=OmegaConf.create(configuration))
    
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        )
    
    print(f'{ctime()} | [{model_label}] Training optimal model...')
    # Train optimized model
    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])
    trainer.fit(tft_model, val_dataloader)

    # Read X and y shapes for CV
    X = batch_val['historical_ts_numeric'].detach().numpy()[:,0,:] # take the first 2d input for cv
    y = batch_val['target'].detach().numpy().flatten()

    # Construct CPCV in-line with DePrado method
    cpcv = CombinatorialPurgedGroupKFold(
        n_splits=10,
        n_test_splits=1,
        embargo_td=pd.Timedelta(days=2)
    )

    cv_split = cpcv.split(
        X=X,
        y=y,
        groups=[e // 10 for e in list(range(len(X)))]
    )

    print(f'{ctime()} | [{model_label}] Running Cross-Validation...')
    # Perform cross-validation
    cv_results = cross_validate_model(
        test_data,
        model=tft_model,
        num_epochs=10,
        num_classes=data_props['num_classes'],
        cv_split=cv_split)
    
    print(f'{ctime()} | [{model_label}] Cross-validation results:')
    print(cv_results)

    print(f'{ctime()} | [{model_label}] Running MC Dropout...')
    # Perform MC Dropout
    mean_predictions, std_predictions = mc_dropout(tft_model, test_dataloader, mc_iterations=10)
    
    print(f'{ctime()} | [{model_label}] Done')
    
    # Gather predicted labels from model
    predicted_labels = np.argmax(mean_predictions, axis=1)

    # Example output with probabilities and uncertainty
    for i, (mean, std) in enumerate(zip(mean_predictions, std_predictions)):
        # softmax_probs = np.exp(mean) / np.sum(np.exp(mean)) # Softmax to get probabilities
        print(f'Sample {i}: Predicted Label = {predicted_labels[i]}, Probabilities = {mean}, Uncertainty (std) = {std}')

    # Save test predictions to a CSV
    test_df = pd.DataFrame({
        'Probability_0': [p[0] for p in mean_predictions],
        'Probability_1': [p[1] for p in mean_predictions],
        'Probability_2': [p[2] for p in mean_predictions],  # Adjust based on num_classes
        'Probability_3': [p[3] for p in mean_predictions],
        'Uncertainty_0': [u[0] for u in std_predictions],
        'Uncertainty_1': [u[1] for u in std_predictions],
        'Uncertainty_2': [u[2] for u in std_predictions],
        'Uncertainty_3': [u[3] for u in std_predictions]
    }, index=test_data[-len(mean_predictions):].index)

    # test_df.to_csv('tft_autoenc_predictions.csv', index=False)
    out_filename = f'{model_label}_test_predictions.csv'

    test_df.to_csv(out_filename, index=False)
    print(f'{ctime()} | [{model_label}] Done, results saved in {out_filename}')

    # Export the model and the data for XAI
    save(tft_model.state_dict(), f'{model_label}.pth')

    torch.save(train_data, f'{model_label}_train_data.pt')
    torch.save(val_data, f'{model_label}_val_data.pt')
    torch.save(test_data, f'{model_label}_test_data.pt')


if __name__ == '__main__':
    tft_classification('1st')