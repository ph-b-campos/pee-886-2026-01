import os
import sys
import pandas as pd
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from data.pedro_campos.config import RANDOM_SEED, BATCH_SIZE, BASE_DIR
from qml.pedro_campos.trainer.lightning import MagicGammaModel
from qml.pedro_campos.loaders.data_loader import MagicGammaDataModule

def kfold_eval(X_df: pd.DataFrame, y_series: pd.Series, base_model_builder_fn, best_params: dict, n_splits: int, train_name: str):
    metricas = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y_series)):
        dm = MagicGammaDataModule(X=X_df, y=y_series, train_index=train_idx, val_index=val_idx, batch_size=BATCH_SIZE)
        
        params_copy = best_params.copy()
        lr = params_copy.pop('lr', 1e-3)
        n_epochs = params_copy.pop('n_epochs', 50)
        
        base_model = base_model_builder_fn(**params_copy)
        lightning_model = MagicGammaModel(model=base_model, learning_rate=lr)
        
        trainer = L.Trainer(max_epochs=n_epochs, enable_progress_bar=False, logger=False, enable_model_summary=False)
        trainer.fit(lightning_model, datamodule=dm)
        
        m = trainer.callback_metrics
        metricas.append({
            'fold': fold + 1,
            'train_loss': m.get('train_loss', torch.tensor(0.0)).item(),
            'val_loss': m.get('val_loss', torch.tensor(0.0)).item(),
            'val_f1': m.get('val_f1', torch.tensor(0.0)).item(),
            'val_auroc': m.get('val_auroc', torch.tensor(0.0)).item()
        })
    
    df_metricas = pd.DataFrame(metricas)
    
    caminho_csv = os.path.join(BASE_DIR, "data", "pedro_campos", f"kfold_{train_name}.csv")
    df_metricas.to_csv(caminho_csv, index=False)

def test_eval(X_tr_df: pd.DataFrame, y_tr_series: pd.Series, X_te_df: pd.DataFrame, y_te_series: pd.Series, base_model_builder_fn, best_params: dict, train_name: str):
    X_full = pd.concat([X_tr_df, X_te_df]).reset_index(drop=True)
    y_full = pd.concat([y_tr_series, y_te_series]).reset_index(drop=True)
    
    train_idx = list(range(len(X_tr_df)))
    test_idx = list(range(len(X_tr_df), len(X_full)))
    
    dm = MagicGammaDataModule(X=X_full, y=y_full, train_index=train_idx, val_index=test_idx, batch_size=BATCH_SIZE)
    dm.setup()
    
    params_copy = best_params.copy()
    lr = params_copy.pop('lr', 1e-3)
    n_epochs = params_copy.pop('n_epochs', 50)
    
    base_model = base_model_builder_fn(**params_copy)
    lightning_model = MagicGammaModel(model=base_model, learning_rate=lr)
    
    logs_dir = os.path.join(BASE_DIR, "data", "pedro_campos", "test_logs")
    logger = CSVLogger(save_dir=logs_dir, name=train_name)
    
    trainer = L.Trainer(max_epochs=n_epochs, enable_progress_bar=False, logger=logger, enable_model_summary=False)
    trainer.fit(lightning_model, datamodule=dm)
    
    lightning_model.eval()
    with torch.no_grad():
        X_te_tensor = dm.val_dataset.tensors[0]
        logits = lightning_model(X_te_tensor).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
    y_te_numpy = y_te_series.to_numpy()
    resultados_teste = {
        'modelo': train_name,
        'accuracy': accuracy_score(y_te_numpy, preds),
        'precision': precision_score(y_te_numpy, preds),
        'recall': recall_score(y_te_numpy, preds),
        'f1_score': f1_score(y_te_numpy, preds),
        'roc_auc': roc_auc_score(y_te_numpy, probs)
    }
    
    df_teste = pd.DataFrame([resultados_teste])
    caminho_teste = os.path.join(BASE_DIR, "data", "pedro_campos", f"teste_{train_name}.csv")
    df_teste.to_csv(caminho_teste, index=False)
    
    df_predicoes = X_te_df.copy()
    df_predicoes['y_real'] = y_te_numpy
    df_predicoes['probabilidade_sinal'] = probs
    df_predicoes['predicao_final'] = preds
    
    caminho_predicoes = os.path.join(BASE_DIR, "data", "pedro_campos", f"predict_{train_name}.csv")
    df_predicoes.to_csv(caminho_predicoes, index=True)