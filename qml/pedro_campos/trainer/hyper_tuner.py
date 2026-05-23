import optuna
import pytorch_lightning as L
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping

from ..loaders.data_loader import MagicGammaDataModule
from ..models.classic import Classifier
from ..models.quantum import QuantumClassifier
from .lightning import MagicGammaModel

def objective_classical(trial, X_train, y_train, skf):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_neurons = trial.suggest_categorical("n_neurons", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    fold_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        data = MagicGammaDataModule(X_train, y_train, train_index, val_index, batch_size=2048)
        model = Classifier(n_neurons, n_layers)
        l_model = MagicGammaModel(model, lr)
        
        early_stop = EarlyStopping(monitor="val_auroc", patience=5, mode="max")
        
        trainer = L.Trainer(
            max_epochs=50, 
            accelerator="cpu",
            devices=1,
            enable_model_summary=False,
            logger=False, 
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[early_stop]
        )
        
        trainer.fit(l_model, data)
        score = trainer.callback_metrics["val_auroc"].item()
        fold_scores.append(score)
        
    return np.mean(fold_scores)

def objective_quantum(trial, X_train, y_train, skf):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    
    fold_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        data = MagicGammaDataModule(X_train, y_train, train_index, val_index, batch_size=128)
        model = QuantumClassifier(n_qubits=6, n_layers=n_layers)
        l_model = MagicGammaModel(model, lr)
        
        early_stop = EarlyStopping(monitor="val_auroc", patience=5, mode="max")
        
        trainer = L.Trainer(
            max_epochs=50,
            accelerator="cpu",
            devices=1,
            enable_model_summary=False,
            logger=False, 
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[early_stop]
        )
        
        trainer.fit(l_model, data)
        score = trainer.callback_metrics["val_auroc"].item()
        fold_scores.append(score)
        
    return np.mean(fold_scores)