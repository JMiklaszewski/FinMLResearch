MODEL_ALIAS = 'TS_Class'
NUM_WORKERS = 1

#------------------------------------
# ENVIRONMENT SETUP
#------------------------------------

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
cur_dir = os.getcwd()
# Add the current directory to system path (including parent folder)
sys.path.append(cur_dir)
sys.path.append(os.path.split(cur_dir)[0])

import numpy as np
import pandas as pd
from datetime import datetime

import keras
from keras import layers
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import optuna
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import classification_report

from time import time, ctime
from CPCV.cpcv import CombinatorialPurgedGroupKFold
import h5py
#------------------------------------
# SAMPLE DATA - REPLACE WITH REAL DATA
#------------------------------------

'Create random multivariate time series (date indexed)'

def generate_time_series(start_date, end_date, num_series):
    date_range = pd.date_range(start=start_date, end=end_date)
    data = np.random.rand(len(date_range), num_series)
    df = pd.DataFrame(data, index=date_range)
    df.columns = [f'Feature_{i}' for i in range(len(df.columns))]
    df.index.name = 'date'
    return df

#------------------------------------
# TS Transformer Model (TF)
#------------------------------------

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def TS_Transformer(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=3
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

#------------------------------------
# HYPERPARAMETER TUNING WITH OPTUNA
#------------------------------------

def ts_transformer_obejctive(trial, x_train, y_train, callbacks, n_classes):

    X, y = x_train, y_train

    # Suggest hyperparameters
    lr = trial.suggest_categorical('lr', [1e-5, 1e-3, 1e-2])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
    head_size = trial.suggest_categorical('head_size', [64, 128, 256])
    dropout_prob = trial.suggest_categorical('dropout_prob', [0.1, 0.3, 0.5])
    hidden_units = trial.suggest_categorical('hidden_units', [[32], [64], [128]])
    filter_size = trial.suggest_categorical('filter_size', [1, 2, 4])
    n_transformer_blocks = trial.suggest_categorical('n_transformer_blocks', [1, 2, 4])
    # num_embedding = trial.suggest_categorical('classifier_units', [16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # Initialize the model with suggested hyperparameters
    model = TS_Transformer(
        input_shape=x_train.shape[1:],
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=filter_size,
        num_transformer_blocks=n_transformer_blocks,
        mlp_units=hidden_units,
        dropout=dropout_prob,
        mlp_dropout=dropout_prob,
        n_classes=n_classes
    )

    # model = TKAT(sequence_length, num_unknow_features, num_know_features, num_embedding, hidden_units, num_heads, n_ahead, use_tkan = True)
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=["sparse_categorical_accuracy"]
        )
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        history = model.fit(
            X_train_fold, y_train_fold, 
            batch_size=batch_size, epochs=10,
            callbacks=callbacks, shuffle=False, verbose = False)

        # Validate the model

        pred_labels = model.predict(X_val_fold)
        true_labels = y_val_fold

        # val_predictions = trainer.predict(model, val_loader)
        # val_predictions = torch.cat([x for x in val_predictions], dim=0).numpy()
        
        val_loss = SparseCategoricalCrossentropy()(true_labels, pred_labels)
        cv_scores.append(val_loss.numpy())

    return np.mean(cv_scores)

#------------------------------------
# CROSS-VALIDATION WITH CPCV
#------------------------------------

def cross_validate_model(X, y, model, cv_split, n_epochs=10, batch_size=16, task_type='classification'):


    fold_results = []
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_split):
        print(f'Fold {fold+1}/{len(list(cv_split))}')

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        history = model.fit(X_train_fold, y_train_fold, 
                            batch_size=batch_size, epochs=n_epochs, 
                            shuffle=False, verbose = False)

        # Validate the model

        pred_labels = model.predict(X_val_fold)
        true_labels = y_val_fold

        if task_type == 'classification':
            all_predictions.append(pred_labels.argmax(axis=1))
        elif task_type == 'regression':
            all_predictions.append(pred_labels)
        # all_probabilities.append(true_labels)
        all_targets.append(true_labels)
    
    # Flatten lists for CSV creation
    all_predictions_flat = [item for sublist in all_predictions for item in sublist]
    # all_probabilities_flat = [item for sublist in all_probabilities for item in sublist]
    all_targets_flat = [item for sublist in all_targets for item in sublist]

    print('Cross-Validation results:')
    if task_type == 'classification':

        # Run classification report
        target_names = [f'Label_{i}' for i in range(pred_labels.shape[1])]
        
        print(classification_report(all_targets_flat, all_predictions_flat, target_names=target_names))
    
    elif task_type == 'regression':

        print(MeanSquaredError()(all_targets_flat, all_predictions_flat))

    return fold_results

#------------------------------------
# MC DROUPOUT
#------------------------------------

def mc_dropout_predictions(model, input_data, num_iterations=100):
    """
    Perform MC Dropout on a compiled TensorFlow model.
    
    Parameters:
    - model: Compiled TensorFlow model with dropout layers.
    - input_data: Input data for which predictions are to be made.
    - num_iterations: Number of MC iterations to perform.
    
    Returns:
    - mean_predictions: Averaged predictions across all iterations.
    - std_predictions: Standard deviation of predictions across all iterations.
    """
    # Create a function to enable dropout during inference
    dropout_model = keras.models.Model(inputs=model.input, outputs=model.output)

    # Collect predictions from each iteration
    predictions = []
    for _ in tqdm(range(num_iterations)):
        # Enable dropout by setting training=True
        preds = dropout_model(input_data, training=True)
        predictions.append(preds.numpy())
    
    # Convert predictions list to a numpy array
    predictions = np.array(predictions)
    
    # Calculate mean and standard deviation across iterations
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    return mean_predictions, std_predictions

#------------------------------------
# MAIN PROGRAM
#------------------------------------

def ts_transformer_classification(model_id):

    model_label = MODEL_ALIAS + '_' + model_id
    print('=' * 50)
    print(f'{ctime()} | [{model_label}] Running...')
    print('=' * 50)

    # Define parameters of multivariate time-series dataframe
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    num_series = 10

    # Generate random time series sample <- PLEASE REPLACE THAT WITH REAL DATA
    df = generate_time_series(start_date, end_date, num_series)

    # Let's switch target column to multiclass labeled column
    df['feature_0'] = np.random.randint(low=0, high=4, size=len(df))

    # Train / Test split <- ALSO REPLACE THAT WITH ACTUAL SPLITS
    x_train, y_train = np.array(df)[:, :-1], np.array(df)[:, -1]
    x_test, y_test = np.array(df)[:, 1:], np.array(df)[:, -1]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Define number of classes
    n_classes = len(np.unique(y_train))
    # Infer input shape
    input_shape = x_train.shape[1:]
    # Define trianing callbacks
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    print(f'{ctime()} | [{model_label}] Optimizing parameters...')

    # Run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: ts_transformer_obejctive(
        trial, x_train, y_train, callbacks, n_classes), n_trials=10)

    # Print best hyperparameters
    print(f"{ctime()} | [{model_label}] Best hyperparameters:", study.best_params)

    # Initialize the model with suggested hyperparameters
    final_model = TS_Transformer(
        input_shape=input_shape,
        head_size=study.best_params['head_size'],
        num_heads=study.best_params['num_heads'],
        ff_dim=study.best_params['filter_size'],
        num_transformer_blocks=study.best_params['filter_size'],
        mlp_units=study.best_params['hidden_units'],
        dropout=study.best_params['dropout_prob'],
        mlp_dropout=study.best_params['dropout_prob'],
        n_classes=n_classes
        )

    # Compile optimized model
    final_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    print(f'{ctime()} | [{model_label}] Training optimal model...')

    # Fit the final model to the training data
    history = final_model.fit(
        x_train, 
        y_train, 
        batch_size=16, 
        epochs=100, 
        validation_split=0.2, 
        callbacks=callbacks, 
        shuffle=False, 
        verbose = True
        )
    
    # Construct CPCV in-line with DePrado method
    cpcv = CombinatorialPurgedGroupKFold(
        n_splits=10,
        n_test_splits=1,
        embargo_td=pd.Timedelta(days=2)
    )

    cv_split = cpcv.split(
        X=x_test,
        y=y_test,
        groups=[e // 10 for e in list(range(len(x_test)))]
    )

    print(f'{ctime()} | [{model_label}] Running Cross-Validation...')
    cv_results = cross_validate_model(
        X = x_test,
        y = y_test,
        model = final_model,
        n_epochs = 10,
        cv_split = cv_split
    )

    print(f'{ctime()} | [{model_label}] Running MC Dropout...')
    # MC Dropout Predictions
    mean_predictions, std_predictions = mc_dropout_predictions(final_model, x_test, num_iterations=100)

    print(f'{ctime()} | [{model_label}] Done!')
    # Export results
    # Save test predictions to a CSV
    test_df = pd.DataFrame({
        'Prediction': mean_predictions.argmax(axis=1),
        'Probability_0': [p[0] for p in mean_predictions],
        'Probability_1': [p[1] for p in mean_predictions],
        'Probability_2': [p[2] for p in mean_predictions],  # Adjust based on num_classes
        'Probability_3': [p[3] for p in mean_predictions],
        'Uncertainty_0': [u[0] for u in std_predictions],
        'Uncertainty_1': [u[1] for u in std_predictions],
        'Uncertainty_2': [u[2] for u in std_predictions],
        'Uncertainty_3': [u[3] for u in std_predictions]
        })

    out_filename = f'{model_label}_test_predictions.csv'
    test_df.to_csv(out_filename, index=False)
    print(f'{ctime()} | [{model_label}] Done, results saved in {out_filename}')

    final_model.save(f'{model_label}.keras')

    with h5py.File(f'{model_label}_data.h5', 'w') as f:
        f.create_dataset('X', data=x_test)
        f.create_dataset('y', data=y_test)

if __name__ == '__main__':
    ts_transformer_classification('1st')