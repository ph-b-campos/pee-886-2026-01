import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

class MagicGammaDataModule(L.LightningDataModule):
    def __init__(self, X, y, train_index, val_index, batch_size=512):
        super().__init__()
        self.X = X
        self.y = y
        self.train_index = train_index
        self.val_index = val_index
        self.batch_size = batch_size
        self.scaler = QuantileTransformer()
        self.pca = PCA(n_components=6)

    def setup(self, stage=None):
        X_train = self.X.iloc[self.train_index]
        X_val = self.X.iloc[self.val_index]
        y_train = self.y.iloc[self.train_index]
        y_val = self.y.iloc[self.val_index]

        self.scaler.fit(X_train)
        X_train_transformed = self.scaler.transform(X_train)
        X_val_transformed = self.scaler.transform(X_val)

        self.pca.fit(X_train_transformed)
        X_train_pca = self.pca.transform(X_train_transformed)
        X_val_pca = self.pca.transform(X_val_transformed)

        tensor_X_train = torch.Tensor(X_train_pca)
        tensor_X_val = torch.Tensor(X_val_pca)
        tensor_y_train = torch.Tensor(y_train.values)
        tensor_y_val = torch.Tensor(y_val.values)

        self.train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
        self.val_dataset = TensorDataset(tensor_X_val, tensor_y_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=False
        )