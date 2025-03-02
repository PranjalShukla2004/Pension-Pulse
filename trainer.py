from models import train_agents
import numpy as np
import config
from functools import lru_cache
import pickle

@lru_cache(maxsize=None)
def get_relationships(bank_id):
    """
    For a given bank agent, vary each input while holding others constant,
    and plot the predicted interest rate.
    """
    agents, _ = train_agents(episodes=config.EPISODES)
    agent = np.random.choice(agents)
   
    # Default state: [wealth, risk_tolerance, loyalty, inflation, gdp_growth, euribor]
    defaults = [500, 0.5, 0.5, 0.02, 0.03, 0.01]
    ranges = {
        0: np.linspace(50, 2000, 100),     # Wealth
        1: np.linspace(0, 1, 100),         # Risk tolerance
        2: np.linspace(0, 1, 100),         # Loyalty
        3: np.linspace(0, 0.1, 100),       # Inflation
        4: np.linspace(0, 0.1, 100),       # GDP growth
        5: np.linspace(0, 0.05, 100)       # EURIBOR
    }
    feature_names = {
        0: 'Wealth',
        1: 'Risk Tolerance',
        2: 'Loyalty',
        3: 'Inflation',
        4: 'GDP Growth',
        5: 'EURIBOR'
    }
    
    final_data = {}
    for i in range(config.STATE_SIZE):
        values = ranges[i]
        predictions = []
        for v in values:
            state = defaults.copy()
            state[i] = v
            pred = agent.predict_rate(state)
            predictions.append(pred)
        
        final_data[feature_names[i]] = (values, predictions)

    return final_data


if __name__ == '__main__':
    relationships = get_relationships(0)
    with open('models/relationships.pkl', 'wb') as f:
        pickle.dump(relationships, f)

