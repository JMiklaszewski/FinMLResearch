import pandas as pd
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from pytorch_lightning.callbacks import ModelCheckpoint

# Prepare sample data
timestamp = pd.date_range(start='2020-01-01', periods=300, freq='D')
time_series = pd.DataFrame({'values': np.random.randn(300)}, index=timestamp)
labels = pd.DataFrame({'label': np.random.randint(0, 3, size=300)}, index=timestamp)
ext_features = pd.DataFrame({
    'feature1': np.random.randn(300),
    'feature2': np.random.randn(300)
}, index=timestamp)

combined_data = time_series.join(labels).join(ext_features)

# Train / Validation / Test Split
train_data = combined_data[:100]
val_data = combined_data[100:200]
test_data = combined_data[200:]

# Read targets
train_target = train_data.label.values
val_target = val_data.label.values
test_target = test_data.label.values

# Read features
train_features = train_data[['feature1', 'feature2']].values
val_features = val_data[['feature1', 'feature2']].values
test_features = test_data[['feature1', 'feature2']].values

# target_val = combined_data.label.values
# ext_features = combined_data[['feature1', 'feature2']].values
# timestamp = combined_data.index.values



# Define a custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, targets, features):
        self.targets = targets
        self.features = features

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.features[idx]

# Define the Autoencoder Classifier model
class AutoencoderClassifier(pl.LightningModule):
    def __init__(self, context_length, num_classes, num_features):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_length + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, context_length),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features):
        # Make sure the dimentions are good to concatenate
        x = x.unsqueeze(1) if x.dim() == 1 else x
        features = features.unsqueeze(1) if features.dim() == 1 else features

        # Concatenate features into one tensor
        x = torch.cat((x, features), dim=1)

        # Pass current tensor through the model
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        class_probs = self.softmax(classification)

        return decoded, class_probs

    def training_step(self, batch, batch_idx):
        targets, features = batch
        decoded, classification = self(targets, features)
        loss = nn.CrossEntropyLoss()(classification, targets.long())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
    
# Instantiate data module and model
data_module = TimeSeriesDataModule(
    train_target, train_features,
    val_target, val_features,
    test_target, test_features,
    batch_size=16
)

# Setup the data for model
data_module.setup()

model = AutoencoderClassifier(context_length=1, num_classes=3, num_features=2)

# Define a checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
)

# Train the model
trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
trainer.fit(model, data_module.train_dataloader())

# Prediction and evaluation
model.eval()
preds, probs = [], []

for batch in data_module.val_dataloader():
    targets, features = batch
    with torch.no_grad():
        decoded, classification = model(targets, features)
        predicted_probs = nn.Softmax(dim=1)(classification)
        predicted_labels = torch.argmax(predicted_probs, dim=1).numpy()
        probs.extend(predicted_probs.numpy())
        preds.extend(predicted_labels)

# Align ground truth and predictions for evaluation
aligned_gt = combined_data.label.iloc[:len(preds)].values

# Classification report
report = classification_report(aligned_gt, preds, target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)

# Show sample output with probabilities
for i, (label, prob) in enumerate(zip(preds, probs)):
    print(f'Sample {i} - Predicted Label: {label}, Probs: {prob}')

' Monte Carlo Dropout - Prediction Uncertainties'

def mc_dropout_predictions(model, data_loader, n_samples=100):
    model.train()  # Ensure dropout is enabled
    predictions = []
    for _ in range(n_samples):
        batch_predictions = []
        for batch in data_loader:
            target, features = batch
            target, features = target.to(model.device), features.to(model.device)
            _, classification = model(target, features)
            batch_predictions.append(classification.detach().cpu().numpy())
        predictions.append(np.concatenate(batch_predictions, axis=0))
    return np.array(predictions)

# Perform MC Dropout predictions
mc_predictions = mc_dropout_predictions(model, data_module.test_dataloader())
# Calculate mean and standard deviation for uncertainty estimates
mean_predictions = mc_predictions.mean(axis=0)
std_predictions = mc_predictions.std(axis=0)
# Convert mean predictions to class labels
predicted_labels = np.argmax(mean_predictions, axis=1)

# Example output with probabilities and uncertainty
for i, (mean, std) in enumerate(zip(mean_predictions, std_predictions)):
    # softmax_probs = np.exp(mean) / np.sum(np.exp(mean)) # Softmax to get probabilities
    print(f'Sample {i}: Predicted Label = {predicted_labels[i]}, Probabilities = {mean}, Uncertainty (std) = {std}')

# Save trained model
torch.save(model.state_dict(), "ae_classifier.pth")

# Function to perform cross-validation
def cross_validate_model(X, y, model_class, context_length, num_classes, num_features, n_splits=5):
    
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

        model = model_class(context_length=context_length, num_classes=num_classes, num_features=num_features)
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
    print("Aggregated Classification Report:")
    print(avg_report)
    return avg_report


# Perform cross-validation
cross_validate_model(val_features, val_target, AutoencoderClassifier, context_length=1, num_classes=3, num_features=2)