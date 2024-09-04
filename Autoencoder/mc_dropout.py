import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iterations=50):
        super().__init__()
        self.model = model
        self.mc_iterations = mc_iterations

    def predict_step(self, batch, batch_idx):
        # Unpack current batch - this needs to be done for each model
        targets, features = batch

        # Enable dropout during inference
        self.model.train()

        if self.model.task == 'classification':
        
            # Perform multiple forward passes
            predictions = [self.model(targets, features)[-1].unsqueeze(0) for _ in range(self.mc_iterations)]

        elif self.model.task == 'regression':
            # Perform multiple forward passes
            predictions = [self.model(targets, features) for _ in range(self.mc_iterations)]
        
        # Stack and average predictions
        predictions = torch.cat(predictions, dim=0)
        mean_predictions = predictions.mean(dim=0)
        std_predictions = predictions.std(dim=0)

        return mean_predictions, std_predictions
    
def calculate_uncertainty(mc_model, data_loader):

    mc_model.model.train()  # Ensure dropout is enabled
    all_mean_predictions = []
    all_std_predictions = []

    # For each batch in data loader
    for batch in tqdm(data_loader):

        # Unpack current batch
        mean_predictions, std_predictions = mc_model.predict_step(batch, None)

        # Collect means and std for current batch
        all_mean_predictions.append(mean_predictions)
        all_std_predictions.append(std_predictions)

    # Join the outputs from all the batches rowwise once done
    all_mean_predictions = torch.cat(all_mean_predictions, dim=0)
    all_std_predictions = torch.cat(all_std_predictions, dim=0)

    return all_mean_predictions, all_std_predictions

def mc_dropout(model, data_loader, mc_iterations=100):
    'This function simplifies running Monte-Carlo dropout to one line of code'

    # Compile model into mc dropout object
    mc_model = LitMCdropoutModel(model, mc_iterations=mc_iterations)

    # Return output of mc dropout
    mean, std = calculate_uncertainty(mc_model, data_loader)

    # Make sure the output is in reusable format (eg. numpy)
    return mean.detach().numpy(), std.detach().numpy()