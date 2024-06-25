import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.regression
import torchvision
import torchmetrics

class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim = 32, depth = 1, task = "regression"):
        super().__init__()

        self.act_fn = nn.ReLU
        self.depth = depth
        layers = [nn.Linear(np.prod(input_dim), hidden_dim), self.act_fn()]
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act_fn()])

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
        
        if task == "classification":
            self.loss = F.binary_cross_entropy_with_logits
            self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=output_dim)
        elif task == "regression":
            self.loss = F.mse_loss
            self.accuracy = torchmetrics.regression.MeanSquaredError()
        else:
            raise Exception("Invalid task")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z, y.flatten().float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return opt
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        val_loss = self.loss(z, y.flatten().float())

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

class asymMLP(MLP):
    def training_step(self, batch, batch_idx):
        # normalize the weight matrix column-wise
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                norms = torch.linalg.norm(layer.weight, axis = 1)
                layer.weight = nn.Parameter((layer.weight / norms[:,None]))
                
        return super().training_step(batch, batch_idx)
    

if __name__ == "__main__":
    M = asymMLP(input_dim=1, output_dim=1, hidden_dim=8, depth=1)
    for layer in M.model:
            if isinstance(layer, nn.Linear):
                norms = torch.linalg.norm(layer.weight, axis = 1)
                print()
                print("Weights shape: ", layer.weight.shape, ", norms shape: ", norms.shape)
                layer.weight = nn.Parameter((layer.weight / norms[:,None]))
                print("Norms before normalization", norms)
                print("Norms after normalization", torch.linalg.norm(layer.weight, axis = 1))
                print()
