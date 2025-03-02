import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import config
from tqdm import tqdm


class BankAgent:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.std_dev = 0.005  # TODO: get from config
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
        """Return the deterministic predicted rate without noise."""
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
            # Scale prediction to rate range
            predicted_mean = predicted * config.MAX_RATE
            # Compute the log probability under a Gaussian with fixed std_dev
            log_probs = -0.5 * tf.square((actions - predicted_mean) / self.std_dev) \
                        - tf.math.log(self.std_dev * tf.sqrt(2 * np.pi))
            # Loss is negative log likelihood weighted by reward (policy gradient objective)
            loss = -tf.reduce_mean(log_probs * rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class BankEnvironment:
    def __init__(self):
        self.client_attributes = np.zeros((config.NUM_CLIENTS, config.LIENT_ATTRIBUTES))
        self.econ_factors = np.zeros(config.ECON_FACTORS)
        
    def reset(self):
        # Generate synthetic clients
        # TODO: get params from config
        self.client_attributes = np.column_stack((
            np.random.lognormal(5, 0.5, config.NUM_CLIENTS),  # Wealth
            np.random.beta(2, 5, config.NUM_CLIENTS),          # Risk tolerance
            np.random.uniform(0, 1, config.NUM_CLIENTS)        # Loyalty
        ))
        # Generate economic factors TODO: get params from config
        self.econ_factors = np.array([
            np.clip(np.random.normal(0.02, 0.005), 0, 0.1),  # Inflation
            np.random.normal(0.03, 0.01)                     # GDP growth TODO: Change GDP to something else
        ])
        return self._get_state()
    
    def _get_state(self):
        avg_client_attributes = np.mean(self.client_attributes, axis=0)  # Shape: (3,)
        econ_factors = self.econ_factors  # Shape: (2,)
        euribor = np.array([np.random.normal(config.EURIBOR_MEAN, config.EURIBOR_VOLATILITY)])  # Shape: (1,)
        # State is a 6D vector: [avg_client_attributes, econ_factors, euribor]
        state = np.concatenate((avg_client_attributes, econ_factors, euribor))
        return state
    
    def calculate_utility(self, rate, agent_id):
        wealth = self.client_attributes[:, 0]
        risk_tol = self.client_attributes[:, 1]
        loyalty = self.client_attributes[:, 2]
        return (rate * 100) + (wealth / 1000) - (risk_tol * 5) + (loyalty * agent_id * 2)


def train_agents(episodes=config.EPISODES):
    env = BankEnvironment()
    agent = BankAgent()
    
    # Pretrain the scaler using synthetic scenarios
    synthetic_scenarios = []
    for _ in tqdm(range(10000), desc="Fitting scaler"):
        scenario = np.concatenate((
            np.random.lognormal(5, 0.5, config.CLIENT_ATTRIBUTES),
            [np.random.normal(0.02, 0.005), np.random.normal(0.03, 0.01)],
            [np.random.normal(0.01, 0.002)]
        ))
        synthetic_scenarios.append(scenario)
    agent.scaler.fit(np.array(synthetic_scenarios))
    
    # Training history for analysis
    history = {
        'rates': [],
        'profits': [],
        'euribor': []
    }
    
    for episode in tqdm(range(episodes), desc="Training episodes"):
        state = env.reset()
        episode_rates = []
        episode_profits = []
        
        for step in range(config.STEPS_PER_EPISODE):
            # Agent selects an interest rate based on the current state
            rate = agent.act(state)
            episode_rates.append(rate)
            
            # Simulate client acquisition
            utilities = env.calculate_utility(rate, 1)  # Assume agent_id = 1
            clients_acquired = np.sum(utilities > 0)  # Count clients with positive utility
            
            # Calculate profit based on the rate chosen and client acquisition
            profit = (config.EURIBOR_MEAN + config.BANK_MARGIN - rate) * clients_acquired * config.DEPOSIT_PER_CLIENT
            episode_profits.append(profit)
            
            # Train the agent on this single step experience
            agent.train([state], [rate], [profit])
            
            # Update state for the next step
            state = env._get_state()
            history['euribor'].append(state[-1])  # Track EURIBOR value
        
        history['rates'].append(np.mean(episode_rates))
        history['profits'].append(np.mean(episode_profits))
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} | Avg Rate: {np.mean(episode_rates):.4f} | Avg Profit: {np.mean(episode_profits):.2f}")
            # Save the model periodically
            agent.model.save(f'models/model_backup{episode+1}.h5')
    
    # Save the final model
    agent.model.save('models/model.h5')
    return agent, history
