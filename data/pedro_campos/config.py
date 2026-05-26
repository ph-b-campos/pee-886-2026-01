import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "pedro_campos", "telescope_data.csv")
DB_FILE_PATH = os.path.join(BASE_DIR, "data", "pedro_campos", "optuna_results.db")
DB_PATH = f"sqlite:///{DB_FILE_PATH}"

RANDOM_SEED = 0
NUM_PCA_COMPONENTS = 6
BATCH_SIZE = 512