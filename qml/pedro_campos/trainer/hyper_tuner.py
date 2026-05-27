import optuna
import pytorch_lightning as L
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping

from ..loaders.data_loader import MagicGammaDataModule
from ..models.classic import Classifier
from ..models.quantum import QuantumClassifier
from .lightning import MagicGammaModel

def objective_classical(trial, X_train, y_train, skf):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_neurons = trial.suggest_categorical("n_neurons", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    fold_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        data = MagicGammaDataModule(X_train, y_train, train_index, val_index, batch_size=2048)
        model = Classifier(n_neurons, n_layers)
        l_model = MagicGammaModel(model, lr)
        
        early_stop = EarlyStopping(monitor="val_f1", patience=10, mode="max")
        
        trainer = L.Trainer(
            max_epochs=50, 
            accelerator="auto", 
            devices=1,
            enable_model_summary=False, 
            logger=False, 
            enable_checkpointing=False,
            enable_progress_bar=False,  
            callbacks=[early_stop]
        )
        
        trainer.fit(l_model, data)
        score = trainer.callback_metrics["val_f1"].item()
        fold_scores.append(score)
        
    return np.mean(fold_scores)

def objective_classical_restricted(trial, X_train, y_train, skf):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_neurons = 6  # Fixo para pareamento justo com os 6 Qubits
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    fold_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        data = MagicGammaDataModule(X_train, y_train, train_index, val_index, batch_size=2048)
        model = Classifier(n_neurons, n_layers)
        l_model = MagicGammaModel(model, lr)
        
        early_stop = EarlyStopping(monitor="val_f1", patience=10, mode="max")
        
        trainer = L.Trainer(
            max_epochs=50, 
            accelerator="auto", 
            devices=1,
            enable_model_summary=False, 
            logger=False, 
            enable_checkpointing=False,
            enable_progress_bar=False,  
            callbacks=[early_stop]
        )
        
        trainer.fit(l_model, data)
        score = trainer.callback_metrics["val_f1"].item()
        fold_scores.append(score)
        
    return np.mean(fold_scores)

def objective_quantum(trial, X_train, y_train, skf):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    
    fold_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        data = MagicGammaDataModule(X_train, y_train, train_index, val_index, batch_size=128)
        
        model = QuantumClassifier(n_qubits=6, n_layers=n_layers) 
        l_model = MagicGammaModel(model, lr)
        
        early_stop = EarlyStopping(monitor="val_f1", patience=10, mode="max")
        
        trainer = L.Trainer(
            max_epochs=50,
            accelerator="auto", 
            devices=1,
            enable_model_summary=False, 
            logger=False, 
            enable_checkpointing=False,
            enable_progress_bar=False,  
            callbacks=[early_stop]
        )
        
        trainer.fit(l_model, data)
        score = trainer.callback_metrics["val_f1"].item()
        fold_scores.append(score)
        
    return np.mean(fold_scores)

def get_best_hyperparameters(db_path: str, study_name: str):
    """
    Conecta-se ao banco de dados do Optuna e extrai os melhores hiperparâmetros de um estudo.
    """
    study = optuna.load_study(study_name=study_name, storage=db_path)
    params = study.best_params
    return params