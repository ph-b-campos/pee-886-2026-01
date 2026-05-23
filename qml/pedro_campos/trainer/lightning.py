import pytorch_lightning as L
import torch
import torch.nn as nn
import torchmetrics

class MagicGammaModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.auroc = torchmetrics.BinaryAUROC()
        self.f1 = torchmetrics.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y.unsqueeze(1).float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y.unsqueeze(1).float())
        
        preds_prob = torch.sigmoid(y_pred)
        auroc_val = self.auroc(preds_prob, y)
        f1_val = self.f1(preds_prob, y)
        
        self.log('val_loss', loss)
        self.log('val_auroc', auroc_val)
        self.log('val_f1', f1_val)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optim