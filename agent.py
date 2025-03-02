import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import config

class BankAgent:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.std_dev = 0.005  # Standard deviation for sampling actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(config.STATE_SIZE,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def act(self, state):
        # Scale state using the fitted StandardScaler
        scaled_state = self.scaler.transform([state])
        # Predict mean rate and scale it to the [0, MAX_RATE] range
        mean_rate = self.model.predict(scaled_state, verbose=0)[0][0] * config.MAX_RATE
        # Sample an action from a Gaussian distribution centered at mean_rate
        action = np.random.normal(mean_rate, self.std_dev)
        return np.clip(action, config.MIN_RATE, config.MAX_RATE)
    
    def predict_rate(self, state):
        """Return the deterministic predicted rate without noise for analysis."""
        scaled_state = self.scaler.transform([state])
        predicted = self.model.predict(scaled_state, verbose=0)[0][0] * config.MAX_RATE
        return predicted
    
    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        scaled_states = self.scaler.transform(states)
        scaled_states = tf.convert_to_tensor(scaled_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predicted = self.model(scaled_states)
            predicted_mean = predicted * config.MAX_RATE  # Scale prediction to rate range
            log_probs = -0.5 * tf.square((actions - predicted_mean) / self.std_dev) \
                        - tf.math.log(self.std_dev * tf.sqrt(2 * np.pi))
            # Policy gradient objective: weight negative log likelihood by reward
            loss = -tf.reduce_mean(log_probs * rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class BankEnvironment:
    def __init__(self):
        self.client_attributes = np.zeros((config.NUM_CLIENTS, config.CLIENT_ATTRIBUTES))
        self.econ_factors = np.zeros(config.ECON_FACTORS)
        
    def reset(self):
        # Generate synthetic client attributes:
        self.client_attributes = np.column_stack((
            np.random.lognormal(5, 0.5, config.NUM_CLIENTS),  # Wealth
            np.random.beta(2, 5, config.NUM_CLIENTS),          # Risk tolerance
            np.random.uniform(0, 1, config.NUM_CLIENTS)        # Loyalty
        ))
        # Generate economic factors:
        self.econ_factors = np.array([
            np.clip(np.random.normal(0.02, 0.005), 0, 0.1),  # Inflation
            np.random.normal(0.03, 0.01)                     # GDP growth
        ])
        return self._get_state()
    
    def _get_state(self):
        avg_client_attributes = np.mean(self.client_attributes, axis=0)  # Shape: (3,)
        econ_factors = self.econ_factors  # Shape: (2,)
        euribor = np.array([np.random.normal(config.EURIBOR_MEAN, config.EURIBOR_VOLATILITY)])  # Shape: (1,)
        # State: [avg_client_attributes, econ_factors, euribor]
        state = np.concatenate((avg_client_attributes, econ_factors, euribor))
        return state
    
    def calculate_utility(self, rate, bank_id):
        """
        Compute customer utility from a bank offering a given rate.
        A penalty is imposed if the rate is below a competitive threshold.
        """
        wealth = self.client_attributes[:, 0]
        risk_tol = self.client_attributes[:, 1]
        loyalty = self.client_attributes[:, 2]
        min_competitive_rate = config.EURIBOR_MEAN  # e.g. 1%
        # Impose penalty if the offered rate is too low
        competitive_penalty = np.where(rate < min_competitive_rate, -50, 0)
        # Utility function: the higher the offered rate, the more attractive it is,
        # but customer attributes and a bank's "reputation" (here represented by bank_id) also play a role.
        return (rate * 100) + (wealth / 1000) - (risk_tol * 5) + (loyalty * bank_id * 2) + competitive_penalty

