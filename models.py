"""
Module for ML models and training pipeline for the  Bank Agent.
"""

from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
import sqlite3
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

import config  

class BankAgent:
    """
    Bank Agent implementing a policy gradient approach.
    """
    def __init__(self) -> None:
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.std_dev: float = 0.005  # Standard deviation for Gaussian sampling of actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    def _build_model(self) -> tf.keras.Model:
        """
        Constructs the neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(config.STATE_SIZE,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def act(self, state: List[float]) -> float:
        """
        Samples an action (interest rate) based on the current state.
        
        :param state: List of state feature values.
        :return: Clipped interest rate in the range [MIN_RATE, MAX_RATE].
        """
        scaled_state = self.scaler.transform([state])
        mean_rate = self.model.predict(scaled_state, verbose=0)[0][0] * config.MAX_RATE
        action = np.random.normal(mean_rate, self.std_dev)
        return float(np.clip(action, config.MIN_RATE, config.MAX_RATE))

    def predict_rate(self, state: List[float]) -> float:
        """
        Returns the deterministic predicted interest rate (without added noise).
        
        :param state: List of state feature values.
        :return: Predicted interest rate.
        """
        scaled_state = self.scaler.transform([state])
        predicted = self.model.predict(scaled_state, verbose=0)[0][0] * config.MAX_RATE
        return float(predicted)

    def train(self, states: List[List[float]], actions: List[float], rewards: List[float]) -> None:
        """
        Updates the agent's model based on a single experience.
        
        :param states: List of state vectors.
        :param actions: List of actions taken.
        :param rewards: List of rewards received.
        """
        states_np = np.array(states)
        actions_np = np.array(actions, dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        scaled_states = self.scaler.transform(states_np)
        scaled_states_tensor = tf.convert_to_tensor(scaled_states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions_np, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predicted = self.model(scaled_states_tensor)
            predicted_mean = predicted * config.MAX_RATE
            log_probs = -0.5 * tf.square((actions_tensor - predicted_mean) / self.std_dev) \
                        - tf.math.log(self.std_dev * tf.sqrt(2 * np.pi))
            loss = -tf.reduce_mean(log_probs * rewards_tensor)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class BankEnvironment:
    """
    Simulation environment that generates synthetic customer and economic data.
    """
    def __init__(self) -> None:
        self.client_attributes = np.zeros((config.NUM_CLIENTS, config.CLIENT_ATTRIBUTES))
        self.econ_factors = np.zeros(config.ECON_FACTORS)

    def reset(self) -> np.ndarray:
        """
        Resets the environment with new synthetic data.
        
        :return: The initial state as a NumPy array.
        """
        self.client_attributes = np.column_stack((
            np.random.lognormal(5, 0.5, config.NUM_CLIENTS),
            np.random.beta(2, 5, config.NUM_CLIENTS),
            np.random.uniform(0, 1, config.NUM_CLIENTS)
        ))
        self.econ_factors = np.array([
            np.clip(np.random.normal(0.02, 0.005), 0, 0.1),
            np.random.normal(0.03, 0.01)
        ])
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Constructs the current state from client attributes, economic factors, and EURIBOR.
        
        :return: State as a NumPy array.
        """
        avg_client_attributes = np.mean(self.client_attributes, axis=0)
        euribor = np.array([np.random.normal(config.EURIBOR_MEAN, config.EURIBOR_VOLATILITY)])
        state = np.concatenate((avg_client_attributes, self.econ_factors, euribor))
        return state

    def calculate_utility(self, rate: float, bank_id: int) -> np.ndarray:
        """
        Computes the per-customer utility for a given offered rate.
        
        :param rate: Offered interest rate.
        :param bank_id: Bank identifier (used as a proxy for reputation).
        :return: NumPy array of utility values per customer.
        """
        wealth = self.client_attributes[:, 0]
        risk_tol = self.client_attributes[:, 1]
        loyalty = self.client_attributes[:, 2]
        min_competitive_rate = config.EURIBOR_MEAN
        competitive_penalty = np.where(rate < min_competitive_rate, -50, 0)
        utility = (rate * 100) + (wealth / 1000) - (risk_tol * 5) + (loyalty * bank_id * 2) + competitive_penalty
        return utility


def train_agents(episodes: int = config.EPISODES) -> Tuple[List[BankAgent], Dict[int, Dict[str, List[float]]]]:
    """
    Trains the bank agents over multiple episodes.
    
    :param episodes: Number of training episodes.
    :return: Tuple of list of trained agents and their training history.
    """
    env = BankEnvironment()
    agents = [BankAgent() for _ in range(config.NUM_BANKS)]
    
    # Pretrain scalers with synthetic scenarios
    synthetic_scenarios = []
    for _ in tqdm(range(10000), desc="Fitting scaler"):
        scenario = np.concatenate((
            np.random.lognormal(5, 0.5, config.CLIENT_ATTRIBUTES),
            [np.random.normal(0.02, 0.005), np.random.normal(0.03, 0.01)],
            [np.random.normal(0.01, 0.002)]
        ))
        synthetic_scenarios.append(scenario)
    synthetic_scenarios = np.array(synthetic_scenarios)
    for agent in agents:
        agent.scaler.fit(synthetic_scenarios)
    
    history: Dict[int, Dict[str, List[float]]] = {i: {'rates': [], 'profits': []} for i in range(config.NUM_BANKS)}
    
    for _ in tqdm(range(episodes), desc="Training episodes"):
        state = env.reset()
        episode_rates = [[] for _ in range(config.NUM_BANKS)]
        episode_profits = [[] for _ in range(config.NUM_BANKS)]
        
        for _ in range(config.STEPS_PER_EPISODE):
            rates = [agent.act(state) for agent in agents]
            utilities = []
            for bank_id, rate in enumerate(rates):
                util = env.calculate_utility(rate, bank_id + 1)
                utilities.append(util)
            utilities = np.array(utilities)
            chosen_bank_indices = np.argmax(utilities, axis=0)
            clients_acquired = [np.sum(chosen_bank_indices == i) for i in range(config.NUM_BANKS)]
            profits = []
            for i in range(config.NUM_BANKS):
                profit = (config.EURIBOR_MEAN + config.BANK_MARGIN - rates[i]) * clients_acquired[i] * config.DEPOSIT_PER_CLIENT
                profits.append(profit)
            
            for i, agent in enumerate(agents):
                agent.train([state], [rates[i]], [profits[i]])
                episode_rates[i].append(rates[i])
                episode_profits[i].append(profits[i])
            
            state = env._get_state()
        
        for i in range(config.NUM_BANKS):
            avg_rate = float(np.mean(episode_rates[i]))
            avg_profit = float(np.mean(episode_profits[i]))
            history[i]['rates'].append(avg_rate)
            history[i]['profits'].append(avg_profit)
    
    # Save final models
    for i, agent in enumerate(agents):
        model_path = config.MODEL_SAVE_PATH_TEMPLATE.format(bank_id=i + 1)
        agent.model.save(model_path)
    
    return agents, history
