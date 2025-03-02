# Hyperparameters and constants
NUM_BANKS = 3
NUM_CLIENTS = 1000
DEPOSIT_PER_CLIENT = 10000
EURIBOR_MEAN = 0.01  # 1% baseline
EURIBOR_VOLATILITY = 0.002
BANK_MARGIN = 0.015  # Bank's return = EURIBOR + margin
EPISODES = 250
STEPS_PER_EPISODE = 24  # Simulating 2 years monthly
MIN_RATE = 0.0
MAX_RATE = 0.05  # 0-5% interest rates
CLIENT_ATTRIBUTES = 3  # wealth, risk_tolerance, loyalty
ECON_FACTORS = 2       # inflation, gdp_growth
STATE_SIZE = 6  # 3 client attributes + 2 economic factors + 1 EURIBOR value
LEARNING_RATE = 0.001