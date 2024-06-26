import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
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
        loss = self.loss(z.flatten().float(), y.flatten().float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return opt
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        val_loss = self.loss(z.flatten().float(), y.flatten().float())

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

class asymMLP(MLP):

    def initialize_weights(self):
        pass

    def __init__(self, input_dim, output_dim, hidden_dim=32, depth=1, task="regression"):
        super().__init__(input_dim, output_dim, hidden_dim, depth, task)
        self.normalize_weights()

    def normalize_weights(self):
        # normalize the weight matrix column-wise
        for i in range(len(self.model) - 1):
            layer = self.model[i]
            next_layer = self.model[i+1]
            if isinstance(layer, nn.Linear) and isinstance(next_layer, nn.ReLU):
                norms = torch.linalg.norm(layer.weight, axis = 1)
                layer.weight = nn.Parameter((layer.weight / norms[:,None]))

    def on_before_zero_grad(self, optimizer: Optimizer):
        self.normalize_weights()
        return super().on_before_zero_grad(optimizer)
    
if __name__ == "__main__":
    M = asymMLP(input_dim=1, output_dim=1, hidden_dim=8, depth=1)
    
    for i in range(len(M.model) - 1):
            layer = M.model[i]
            next_layer = M.model[i+1]
            if isinstance(layer, nn.Linear) and isinstance(next_layer, nn.ReLU):
                norms = torch.linalg.norm(layer.weight, axis = 1)
                print()
                print("Weights shape: ", layer.weight.shape, ", norms shape: ", norms.shape)
                layer.weight = nn.Parameter((layer.weight / norms[:,None]))
                # print("Norms before normalization", norms)
                # print("Norms after normalization", torch.linalg.norm(layer.weight, axis = 1))
                print()
