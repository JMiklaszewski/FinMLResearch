import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


# Define the Autoencoder Classifier model
class AutoencoderClassifier(pl.LightningModule):
    def __init__(self, context_length, num_classes, num_features, dropout=0.5):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_length + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout) # Apply dropout after activation
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, context_length),
            nn.ReLU() # No dropout here, to not deteriorate decoding ability
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout), # Apply dropout after activation
            nn.Linear(32, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features=None):
        if features is not None:
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

        # Unpack current batch to target/features
        targets, features = batch
        
        # Get model output for current batch
        decoded, classification = self(targets, features)

        # Record current loss
        loss = nn.CrossEntropyLoss()(classification, targets.long())
        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

class AutoencoderAttentionClassifier(pl.LightningModule):
    def __init__(self, context_length, num_classes, num_features, num_heads=2, dropout_prob=0.5, hidden_units=128, embed_dim=64, classifier_units=32, lr=1e-3):
        super(AutoencoderAttentionClassifier, self).__init__()
        self.save_hyperparameters()

        # First layer - encoder
        self.encoder = nn.Sequential(
            nn.Linear(context_length + num_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_units, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Second layer - Multi Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=dropout_prob, # Regularizes attention mechanism
            batch_first=True
            )
        
        # Third layer - decoder (linear output)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, context_length),
            nn.ReLU() # No dropout to not decrease decoding performance
        )

        # Additional - classification layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(classifier_units, num_classes)
        )

        # Softmax activation for logits to probability transformation
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr

    def forward(self, x, features=None):
        # If features have been passed separately
        if features is not None:

            # Make sure the dimentions are good to concatenate
            x = x.unsqueeze(1) if x.dim() == 1 else x
            features = features.unsqueeze(1) if features.dim() == 1 else features
        
            # Concatenate features into one tensor
            x = torch.cat((x, features), dim=1)

        # If inputs is passed as numpy array (LIME) - convert it back to tensor
        if type(x) == np.ndarray:
            x = torch.tensor(x)

        # Pass current tensor through the model
        encoded = self.encoder(x)
        encoded = encoded.unsqueeze(1)
        attn_output, _ = self.attention(encoded, encoded, encoded)
        attn_output = attn_output.squeeze(1)
        decoded = self.decoder(attn_output)
        classification = self.classifier(attn_output)
        class_probs = self.softmax(classification)

        if features is not None:
            return decoded, class_probs
        else: 
            return class_probs
        # return decoded, class_probs if features is not None else class_probs

    def training_step(self, batch, batch_idx):

        # Unpack current batch to targets/features
        targets, features = batch
        
        # Get model output for current batch
        decoded, classification = self(targets, features)
        
        # Record current loss
        loss = nn.CrossEntropyLoss()(classification, targets.long())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)