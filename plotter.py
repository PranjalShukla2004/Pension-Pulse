import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_2d_relationships(agent, bank_id=1):
    """
    For a given bank agent, vary each input while holding others constant,
    and plot the predicted interest rate.
    """
    # Default state: [wealth, risk_tolerance, loyalty, inflation, gdp_growth, euribor]
    defaults = [500, 0.5, 0.5, 0.02, 0.03, 0.01]
    ranges = {
        0: np.linspace(50, 2000, 100),   # Wealth
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
    
    for i in range(config.STATE_SIZE):
        values = ranges[i]
        predictions = []
        for v in values:
            state = defaults.copy()
            state[i] = v
            pred = agent.predict_rate(state)
            predictions.append(pred)
        
        plt.figure(figsize=(8, 5))
        plt.plot(values, predictions, color='blue', linewidth=2)
        plt.xlabel(feature_names[i])
        plt.ylabel('Predicted Interest Rate')
        plt.title(f'Bank {bank_id}: {feature_names[i]} vs. Interest Rate')
        plt.grid(True)
        plt.savefig(f'2d_plot_bank_{bank_id}_{feature_names[i].lower().replace(" ", "_")}.png')
        plt.show()


# Load the trained agents
with open('agents.pkl', 'rb') as f:
    agents = pickle.load(f)
    for i, agent in enumerate(agents):
        plot_2d_relationships(agent, bank_id=i+1)
