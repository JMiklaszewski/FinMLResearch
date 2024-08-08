import torch
import torch.nn as nn
import pytorch_lightning as pl


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
    

class AutoencoderAttentionClassifier(pl.LightningModule):
    def __init__(self, context_length, num_classes, num_features, num_heads=2, dropout_prob=0.5, hidden_units=128, embed_dim=64, classifier_units=32, lr=1e-3):
        super(AutoencoderAttentionClassifier, self).__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(context_length + num_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_units, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_units, context_length),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(classifier_units, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)
        self.lr = lr

    def forward(self, x, features):
        x = x.unsqueeze(1) if x.dim() == 1 else x
        features = features.unsqueeze(1) if features.dim() == 1 else features
        x = torch.cat((x, features), dim=1)
        encoded = self.encoder(x)
        encoded = encoded.unsqueeze(1)
        attn_output, _ = self.attention(encoded, encoded, encoded)
        attn_output = attn_output.squeeze(1)
        decoded = self.decoder(attn_output)
        classification = self.classifier(attn_output)
        class_probs = self.softmax(classification)
        return decoded, class_probs

    def training_step(self, batch, batch_idx):
        targets, features = batch
        decoded, classification = self(targets, features)
        loss = nn.CrossEntropyLoss()(classification, targets.long())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)