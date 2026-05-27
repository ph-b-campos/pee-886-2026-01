import pandas as pd
import optuna
import torch
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.pedro_campos.config import DATA_PATH, RANDOM_SEED, DB_PATH
from qml.pedro_campos.trainer.hyper_tuner import objective_classical, objective_classical_restricted, objective_quantum

def main():
    df = pd.read_csv(DATA_PATH) 
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'g': 1, 'h': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_SEED
    )
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    
    study_classic = optuna.create_study(
        direction="maximize", 
        study_name="MLP_Baseline",
        storage=DB_PATH,
        load_if_exists=True 
    )
    study_classic.optimize(
        lambda trial: objective_classical(trial, X_train, y_train, skf), 
        n_trials=15
    )
    
    study_classic_restricted = optuna.create_study(
        direction="maximize", 
        study_name="MLP_Restricted",
        storage=DB_PATH,
        load_if_exists=True 
    )
    study_classic_restricted.optimize(
        lambda trial: objective_classical_restricted(trial, X_train, y_train, skf), 
        n_trials=10
    )
    
    study_quantum = optuna.create_study(
        direction="maximize", 
        study_name="QNN_Ansatz",
        storage=DB_PATH,
        load_if_exists=True 
    )
    study_quantum.optimize(
        lambda trial: objective_quantum(trial, X_train, y_train, skf), 
        n_trials=10 
    )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()