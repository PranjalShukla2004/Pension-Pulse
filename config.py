"""
Configuration file for the Advanced Bank Agent project.
All hyperparameters, environment settings, and file path templates are defined here.
"""

import numpy as np

# Environment Parameters
NUM_BANKS: int = 5
NUM_CLIENTS: int = 1000
CLIENT_ATTRIBUTES: int = 3        # e.g., wealth, risk_tolerance, loyalty
ECON_FACTORS: int = 2             # e.g., inflation, gdp_growth
STATE_SIZE: int = 6               # 3 client attributes + 2 economic factors + 1 EURIBOR value

# Bank Parameters
DEPOSIT_PER_CLIENT: int = 1000    # €1000 per client
EURIBOR_MEAN: float = 0.01        # 1% baseline rate
EURIBOR_VOLATILITY: float = 0.002
BANK_MARGIN: float = 0.015        # Bank's return = EURIBOR + margin
MIN_RATE: float = 0.0
MAX_RATE: float = 0.05            # Interest rates in the range [0, 5%]

# Training Parameters
EPISODES: int = 50
STEPS_PER_EPISODE: int = 24       # Simulating 2 years with monthly steps
LEARNING_RATE: float = 0.001

# Prediction Grid Parameters (for saving model predictions to DB)
DEFAULT_STATE: list = [500, 0.5, 0.5, 0.02, 0.03, 0.01]
PREDICTION_GRID_RANGES: dict = {
    'wealth': np.linspace(50, 2000, 50),
    'risk_tolerance': np.linspace(0, 1, 50),
    'loyalty': np.linspace(0, 1, 50),
    'inflation': np.linspace(0, 0.1, 50),
    'gdp_growth': np.linspace(0, 0.1, 50),
    'euribor': np.linspace(0, 0.05, 50)
}

# SQLite Database Configuration
PREDICTIONS_DB_PATH_TEMPLATE: str = 'models/predictions_bank_{bank_id}.db'

# Model Save Path Template
MODEL_SAVE_PATH_TEMPLATE: str = '/models/advanced_bank_agent_bank_{bank_id}_final.h5'
