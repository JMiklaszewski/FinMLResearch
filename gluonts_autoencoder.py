import pandas as pd
import numpy as np

from gluonts.dataset.common import ListDataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.field_names import FieldName

timestamp = pd.date_range(start='2020-01-01', periods=100, freq='D')

'Prepare sample data that includes time series, external features and labels'
time_series = pd.DataFrame({'values': np.random.randn(100)}, index=timestamp)

labels = pd.DataFrame({'label': np.random.randint(0, 3, size=100)}, index=timestamp)

ext_features = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100)
}, index=timestamp)

# Combine data into one dataset
combined_data = time_series.join(labels).join(ext_features)

# Prepare ListDataset
target_val = combined_data.label.values
ext_features = combined_data[['feature1', 'feature2']].values.T
timestamp = combined_data.index.values

' Convert data format to GluonTS ListDataset'
ds = ListDataset(
    [{
        FieldName.TARGET: target_val, 
        FieldName.START: timestamp[0], 
        FieldName.FEAT_DYNAMIC_REAL: ext_features
        
    }], 
        freq='D')


' Define Custom Autoencoder Model'
import mxnet as mx
from mxnet.gluon import nn, Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.data import ArrayDataset, DataLoader

class AutoencoderClassifier(nn.Block):

    def __init__(self, context_length, num_classes, num_features, **kwargs):
        # Run parent class initializer
        super().__init__(**kwargs)

        # Define encoder part of network
        self.encoder = nn.Sequential()
        self.encoder.add(nn.Dense(units=128, activation='relu'))
        self.encoder.add(nn.Dense(units=64, activation='relu'))

        # Define decoder part of network
        self.decoder = nn.Sequential()
        self.decoder.add(nn.Dense(units=128, activation='relu'))
        self.decoder.add(nn.Dense(units=context_length, activation='relu')) # Number of output units have to match our context length

        # Define classifier head of the model
        self.classifier = nn.Sequential()
        self.classifier.add(nn.Dense(units=32, activation='relu'))
        self.classifier.add(nn.Dense(num_classes)) # Number of output neurons has to match our number of classes (1 probability per 1 class)
        #self.classifier.add(mx.sym.SoftmaxOutput()) # We add softmax to get probabilities for each class

    def forward(self, x, features):

        x = mx.nd.concat(x, features, dim=1)
        # Run the encoder part of the network
        encoded = self.encoder(x)
        # Now, we decode the same data with decoder
        decoded = self.decoder(encoded)
        # Perform classification of encoded data
        classification = self.classifier(encoded)

        # Apply softmax function so that proper class probabilities are output
        class_probs = mx.nd.softmax(classification)

        return decoded, class_probs # Return both forward predictions as well as class probabilities
    
# Define mx context
context = mx.cpu()

# Compile the model
model = AutoencoderClassifier(context_length=20, num_classes=3, num_features=2)

# Initialize the model
model.initialize(mx.init.Xavier(), ctx=context)

'Prepare the dataset'
data = ArrayDataset(mx.nd.array(target_val).expand_dims(axis=1), 
                    mx.nd.array(ext_features.T))

# Specify how to load the data
data_loader = DataLoader(data, batch_size=16, shuffle=False)

'Train the model'

# Define the trainer and loss function
trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-3})
loss_fn = SoftmaxCrossEntropyLoss()

# Training loop
from tqdm import tqdm

n_epochs = 50
for epoch in tqdm(range(n_epochs)):
    cum_loss = 0

    for batch in data_loader:
        with mx.autograd.record():
            target = batch[0].as_in_context(context)
            features = batch[1].as_in_context(context)
            decoded, classification = model(target, features)
            loss = loss_fn(classification, target.squeeze())
        
        loss.backward()
        trainer.step(batch_size=16, ignore_stale_grad=True)
        cum_loss += mx.nd.sum(loss).asscalar()
    
    print(f'Epoch {epoch+1}, Loss: {cum_loss}')


'Prediction and Evaluation'
preds = []
probs = []

for batch in data_loader:
    target = batch[0].as_in_context(context)
    features = batch[1].as_in_context(context)
    decoded, classification = model(target, features)

    predicted_probs = classification # save the probabilities for each class
    predicted_labels = np.argmax(predicted_probs, axis=1).asnumpy() # 'select' the most probable class from softmax
    
    probs.extend(predicted_probs.asnumpy())
    preds.extend(predicted_labels)


# Align ground truth and predictions for evaluation
aligned_gt = combined_data.label.iloc[:len(preds)].values

# Classification report (from sklearn)
from sklearn.metrics import classification_report
report = classification_report(aligned_gt, preds,
                               target_names=['Class 0', 'Class 1', 'Class 2'])

print(report)

# Show sample output with probabilities
for i, (label, prob) in enumerate(zip(preds, probs)):
    print(f'Sample {i} - Predicted Label: {label}, Probs: {prob}')


'Qunatifying Uncertainty - Monte Carlo Dropout'

# 1. First, we add a dropout option to our model
class AutoencoderWithClassifier(nn.Block):
    def __init__(self, context_length, num_classes, num_features,
                 dropout_rate=0.5, **kwargs):
         super().__init__(**kwargs)
         self.encoder = nn.Sequential()
         self.encoder.add(nn.Dense(128, activation='relu'))
         self.encoder.add(nn.Dropout(dropout_rate)) # Add dropout
         self.encoder.add(nn.Dense(64, activation='relu'))
         self.encoder.add(nn.Dropout(dropout_rate)) # Add dropout
         
         self.decoder = nn.Sequential()
         self.decoder.add(nn.Dense(128, activation='relu'))
         self.decoder.add(nn.Dense(context_length, activation='relu'))
         
         self.classifier = nn.Sequential()
         self.classifier.add(nn.Dense(32, activation='relu'))
         self.classifier.add(nn.Dropout(dropout_rate)) # Add dropout
         self.classifier.add(nn.Dense(num_classes))
         # self.classifier.add(nn.SoftmaxActivation()) # Add Softmax for probabilities
    
    def forward(self, x, features):

        x = mx.nd.concat(x, features, dim=1)
        # Run the encoder part of the network
        encoded = self.encoder(x)
        # Now, we decode the same data with decoder - this is our prediction of next label
        decoded = self.decoder(encoded)
        # Perform classification of encoded data - this is our prediction of t+1 probabilities
        classification = self.classifier(encoded)

        # Apply softmax function so that proper class probabilities are output
        class_probs = mx.nd.softmax(classification)

        return decoded, class_probs # Return both forward predictions as well as class probabilities
    

# Define context
context = mx.cpu()
# Define the model
model = AutoencoderWithClassifier(context_length=20, num_classes=3,
num_features=2, dropout_rate=0.5)
model.initialize(mx.init.Xavier(), ctx=context)

# Define the trainer
trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-3})
loss_fn = SoftmaxCrossEntropyLoss()

# Training loop
for epoch in range(50):
    cumulative_loss = 0
    for batch in data_loader:
        with mx.autograd.record():
            target = batch[0].as_in_context(context)
            features = batch[1].as_in_context(context)
            decoded, classification = model(target, features)
            loss = loss_fn(classification, target.squeeze())
        
        loss.backward()
        trainer.step(batch_size=16, ignore_stale_grad=True)
        cumulative_loss += mx.nd.sum(loss).asscalar()
    
    print(f'[MC] Epoch {epoch + 1}, Loss: {cumulative_loss}')

def mc_dropout_predictions(model, data_loader, n_samples=100):
    # model.train() # Ensure dropout is enabled
    predictions = []
    for _ in range(n_samples):
        batch_predictions = []
        for batch in data_loader:
            target = batch[0].as_in_context(context)
            features = batch[1].as_in_context(context)
            _, classification = model(target, features)
            batch_predictions.append(classification.asnumpy())
        
        predictions.append(np.concatenate(batch_predictions, axis=0))
        
    
    return np.array(predictions)

# Perform MC Dropout predictions
mc_predictions = mc_dropout_predictions(model, data_loader)
# Calculate mean and standard deviation for uncertainty estimates
mean_predictions = mc_predictions.mean(axis=0)
std_predictions = mc_predictions.std(axis=0)
# Convert mean predictions to class labels
predicted_labels = np.argmax(mean_predictions, axis=1)

from sklearn.metrics import classification_report
# Align ground truth and predictions for evaluation
aligned_truth = combined_data['label'].iloc[:len(predicted_labels)].values
# Classification report
report = classification_report(aligned_truth, predicted_labels,
target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)

# Example output with probabilities and uncertainty
for i, (mean, std) in enumerate(zip(mean_predictions, std_predictions)):
    softmax_probs = np.exp(mean) / np.sum(np.exp(mean)) # Softmax to get probabilities
    print(f'Sample {i}: Predicted Label = {predicted_labels[i]}, Probabilities = {softmax_probs}, Uncertainty (std) = {std}')