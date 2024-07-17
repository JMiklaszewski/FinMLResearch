import pandas as pd
import numpy as np
from functools import reduce
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

# Define a custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, targets, features):
        self.targets = targets
        self.features = features

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.features[idx]
    
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, train_target, train_features, val_target, val_features, test_target, test_features, batch_size=16):
        super().__init__()
        self.train_target = train_target
        self.train_features = train_features
        self.val_target = val_target
        self.val_features = val_features
        self.test_target = test_target
        self.test_features = test_features
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TimeSeriesDataset(
            torch.tensor(self.train_target, dtype=torch.float32),
            torch.tensor(self.train_features, dtype=torch.float32)
        )
        
        self.val_dataset = TimeSeriesDataset(
            torch.tensor(self.val_target, dtype=torch.float32),
            torch.tensor(self.val_features, dtype=torch.float32)
        )

        self.test_dataset = TimeSeriesDataset(
            torch.tensor(self.test_target, dtype=torch.float32),
            torch.tensor(self.test_features, dtype=torch.float32)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

' Monte Carlo Dropout - Prediction Uncertainties'

def mc_dropout_predictions(model, data_loader, n_samples=100, num_classes=3):
    model.train()  # Ensure dropout is enabled
    predictions = []
    for _ in range(n_samples):
        batch_predictions = []
        for batch in data_loader:
            if len(batch) == 2:
                target, features = batch
                target, features = target.to(model.device), features.to(model.device)
                _, classification = model(target, features)
            else:
                static_inputs, past_inputs, future_inputs, targets = batch
                outputs = model(static_inputs, past_inputs, future_inputs)
                logits = outputs[:, :, :num_classes]  # Extract logits
                probabilities = outputs[:, :, num_classes:] # Extract probabilities
                
                logits = torch.reshape(logits, (logits.shape[0], logits.shape[-1]))

                classification = logits
            batch_predictions.append(classification.detach().cpu().numpy())
        predictions.append(np.concatenate(batch_predictions, axis=0))
    return np.array(predictions)

# Function to perform cross-validation
def cross_validate_model(X, y, model_class, context_length, num_classes, num_features, n_splits=5, num_heads=2, dropout_prob=0.5, hidden_units=128, embed_dim=64, classifier_units=32, lr=1e-3):
    
    # # Prepare tensors
    # tensor_x = torch.tensor(X)
    # tensor_y = torch.LongTensor(y)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aggregated_report = []

    for train_index, test_index in tscv.split(X):
        train_data = DataLoader(TimeSeriesDataset(
            torch.tensor(y[train_index], dtype=torch.float32), 
            torch.tensor(X[train_index], dtype=torch.float32)), batch_size=16)
        
        val_data = DataLoader(TimeSeriesDataset(
            torch.tensor(y[test_index], dtype=torch.float32), 
            torch.tensor(X[test_index], dtype=torch.float32)), batch_size=16)

        model = model_class(
            context_length=context_length, 
            num_classes=num_classes, 
            num_features=num_features,
            num_heads = num_heads,
            dropout_prob=dropout_prob, 
            hidden_units=hidden_units, 
            embed_dim=embed_dim, 
            classifier_units=classifier_units, 
            lr=lr)
        
        trainer = pl.Trainer(max_epochs=50)
        trainer.fit(model, train_dataloaders=train_data)

        # Make predictions on the validation set
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_data:
                targets, features = batch
                _, classification = model(targets, features)
                preds = torch.argmax(classification, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute classification report
        report = classification_report(all_targets, all_preds, output_dict=True)
        aggregated_report.append(report)
    
    # Aggregate classification reports
    reports = [pd.DataFrame(i) for i in aggregated_report]
    avg_report = reduce(lambda x,y: x.add(y, fill_value=0), reports)/len(reports)

    print("Cross-Validation Classification Report:")
    print(avg_report)
    
    return avg_report