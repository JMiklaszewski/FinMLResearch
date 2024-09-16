MODEL_ALIAS = 'AE_Class'
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
sys.path.append(os.path.split(cur_dir)[0])

import pandas as pd
import numpy as np

from pl_autoencoder_classifiers import AutoencoderAttentionClassifier
from pl_model_utils import TimeSeriesDataModule, cross_validate_model, TimeSeriesDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from mc_dropout import mc_dropout
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss
import optuna
from tqdm import tqdm
from torch import save
import h5py
from time import time, ctime

from CPCV.cpcv import CombinatorialPurgedGroupKFold
from sklearn.model_selection import TimeSeriesSplit

# PLEASE MAKE SURE WE USE SEED FOR REPRODUCABILITY
seed_everything(123, workers=True)

from lightning.pytorch.utilities import disable_possible_user_warnings
disable_possible_user_warnings()
#------------------------------------
# SAMPLE DATA - REPLACE WITH REAL DATA
#------------------------------------

def generate_random_data():
    # Prepare sample data
    timestamp = pd.date_range(start='2020-01-01', periods=300, freq='D')
    time_series = pd.DataFrame({'values': np.random.randn(300)}, index=timestamp)
    labels = pd.DataFrame({'label': np.random.randint(0, 3, size=300)}, index=timestamp)
    ext_features = pd.DataFrame({
        'feature1': np.random.randn(300),
        'feature2': np.random.randn(300)
        }, index=timestamp)

    return time_series.join(labels).join(ext_features)

#------------------------------------
# HYPERPARAMETER TUNING WITH OPTUNA
#------------------------------------

def ae_attention_objective(trial):
    context_length = 1
    num_classes = 3
    num_features = 2

    # Suggest hyperparameters
    lr = trial.suggest_categorical('lr', [1e-5, 1e-3, 1e-2])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    dropout_prob = trial.suggest_categorical('dropout_prob', [0.1, 0.3, 0.5])
    hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
    embed_dim = trial.suggest_categorical('embed_dim', [32, 64, 128])
    classifier_units = trial.suggest_categorical('classifier_units', [16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Initialize the model with suggested hyperparameters
    model = AutoencoderAttentionClassifier(
        context_length=context_length,
        num_classes=num_classes,
        num_features=num_features,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        hidden_units=hidden_units,
        embed_dim=embed_dim,
        classifier_units=classifier_units,
        lr=lr,
        task='classification'
    )

    # Assuming you have your dataset in `X` and `y`
    X, y = data_module.val_features, data_module.val_target
    # X = np.array(X_train)  # Ensure X_train is a NumPy array
    # y = np.array(y_train)  # Ensure y_train is a NumPy array

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_index, val_index in tqdm(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Create DataLoader for the training and validation fold
        train_dataset = TimeSeriesDataset(
            torch.tensor(y_train_fold, dtype=torch.float32), 
            torch.tensor(X_train_fold, dtype=torch.float32)
            )
        
        val_dataset = TimeSeriesDataset(
            torch.tensor(y_val_fold, dtype=torch.float32), 
            torch.tensor(X_val_fold, dtype=torch.float32)
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize trainer - switch all of the optional prints please
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[EarlyStopping(monitor='train_loss', patience=3, mode='min')],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False
        )

        # Train the model
        trainer.fit(model, train_loader)

        # Validate the model

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                targets, features = batch
                _, classification = model(targets, features)
                # preds = torch.argmax(classification, dim=1)
                all_preds.extend(classification.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # val_predictions = trainer.predict(model, val_loader)
        # val_predictions = torch.cat([x for x in val_predictions], dim=0).numpy()
        
        val_loss = log_loss(all_targets, all_preds)
        cv_scores.append(val_loss)

    return np.mean(cv_scores) 

def ae_classification(model_id):
    model_label = MODEL_ALIAS + '_' + model_id
    print('=' * 50)
    print(f'{ctime()} | [{model_label}] Running...')
    print('=' * 50)


    # Specify input - make sure to use real data
    df = generate_random_data()

    # Train / Validation / Test Split
    train_data = df[:100]
    val_data = df[100:200]
    test_data = df[200:]

    # Read targets
    train_target = train_data.label.values
    val_target = val_data.label.values
    test_target = test_data.label.values

    # Read features
    train_features = train_data[['feature1', 'feature2']].values
    val_features = val_data[['feature1', 'feature2']].values
    test_features = test_data[['feature1', 'feature2']].values

    # Instantiate data module and model
    global data_module
    data_module = TimeSeriesDataModule(
        train_target, train_features,
        val_target, val_features,
        test_target, test_features,
        batch_size=16
    )

    # Setup the data for model
    data_module.setup()

    print(f'{ctime()} | [{model_label}] Optimizing parameters...')
    # Run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(ae_attention_objective, n_trials=10)

    # Print best hyperparameters
    print(f"{ctime()} | [{model_label}] Best hyperparameters:", study.best_params)

    # Speciy the Autoencoder Model with optmial hyperparameters
    ae_attention_model = AutoencoderAttentionClassifier(
        context_length=1, 
        num_classes=3, 
        num_features=2,
        lr=study.best_params['lr'],
        num_heads=study.best_params['num_heads'],
        dropout_prob=study.best_params['dropout_prob'],
        hidden_units=study.best_params['hidden_units'],
        embed_dim=study.best_params['embed_dim'],
        classifier_units=study.best_params['classifier_units'],
        task='classification'
        )
    
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
    trainer.fit(ae_attention_model, data_module.train_dataloader())

    # Construct CPCV in-line with DePrado method
    cpcv = CombinatorialPurgedGroupKFold(
        n_splits=10,
        n_test_splits=1,
        embargo_td=pd.Timedelta(days=2)
    )

    cv_split = cpcv.split(
        X=train_features,
        y=train_target,
        groups=[e // 10 for e in list(range(len(train_features)))]
    )

    print(f'{ctime()} | [{model_label}] Running Cross-Validation...')
    # Perform cross-validation
    cv_results = cross_validate_model(
        X=data_module.val_features, 
        y=data_module.val_target, 
        model=ae_attention_model,
        cv_split=cv_split)
    
    print(f'{ctime()} | [{model_label}] Cross-validation results:')
    print(cv_results)

    print(f'{ctime()} | [{model_label}] Running MC Dropout...')
    # Perform MC Dropout
    predictions_mean, predictions_std = mc_dropout(
        ae_attention_model, 
        data_module.val_dataloader(), 
        mc_iterations=100
        )
    
    print(f'{ctime()} | [{model_label}] Done')
    
    # Gather predicted labels from model
    predicted_labels = np.argmax(predictions_mean, axis=1)

    # Example output with probabilities and uncertainty
    for i, (mean, std) in enumerate(zip(predictions_mean, predictions_std)):
        # softmax_probs = np.exp(mean) / np.sum(np.exp(mean)) # Softmax to get probabilities
        print(f'Sample {i}: Predicted Label = {predicted_labels[i]}, Probabilities = {mean}, Uncertainty (std) = {std}')

    # Save test predictions to a CSV
    test_df = pd.DataFrame({
        'Prediction': predicted_labels,
        'Probability_0': [p[0] for p in predictions_mean],
        'Probability_1': [p[1] for p in predictions_mean],
        'Probability_2': [p[2] for p in predictions_mean],  # Adjust based on num_classes
        'Uncertainty_0': [u[0] for u in predictions_std],
        'Uncertainty_1': [u[1] for u in predictions_std],
        'Uncertainty_2': [u[2] for u in predictions_std] 
        })
    
    out_filename = f'{model_label}_test_predictions.csv'

    test_df.to_csv(out_filename, index=False)
    print(f'{ctime()} | [{model_label}] Done, results saved in {out_filename}')

    save(ae_attention_model.state_dict(), f'{model_label}.pth')

    with h5py.File(f'{model_label}_data.h5', 'w') as f:
        f.create_dataset('X', data=data_module.val_features)
        f.create_dataset('y', data=data_module.val_target)

if __name__ == '__main__':
    ae_classification('1st')