import os
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.pedro_campos.config import DATA_PATH, RANDOM_SEED, DB_PATH,NUM_PCA_COMPONENTS
from qml.pedro_campos.trainer.hyper_tuner import get_best_hyperparameters
from qml.pedro_campos.evaluation.eval import kfold_eval, test_eval
from qml.pedro_campos.models.classic import Classifier
from qml.pedro_campos.models.quantum import QuantumClassifier

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'g': 1, 'h': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_SEED
    )
    
    best_params_mlp = get_best_hyperparameters(
        db_path=DB_PATH, 
        study_name="MLP_Baseline"
    )
    
    kfold_eval(
        X_df=X_train, 
        y_series=y_train, 
        base_model_builder_fn=Classifier, 
        best_params=best_params_mlp, 
        n_splits=10, 
        train_name="mlp"
    )
    
    test_eval(
        X_tr_df=X_train, 
        y_tr_series=y_train, 
        X_te_df=X_test, 
        y_te_series=y_test, 
        base_model_builder_fn=Classifier, 
        best_params=best_params_mlp, 
        train_name="mlp"
    )
    
    best_params_qnn = get_best_hyperparameters(
        db_path=DB_PATH, 
        study_name="QNN_Ansatz"
    )
    best_params_qnn['n_qubits'] = NUM_PCA_COMPONENTS


    
    kfold_eval(
        X_df=X_train, 
        y_series=y_train, 
        base_model_builder_fn=QuantumClassifier, 
        best_params=best_params_qnn, 
        n_splits=10, 
        train_name="qnn"
    )
    
    test_eval(
        X_tr_df=X_train, 
        y_tr_series=y_train, 
        X_te_df=X_test, 
        y_te_series=y_test, 
        base_model_builder_fn=QuantumClassifier, 
        best_params=best_params_qnn, 
        train_name="qnn"
    )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()