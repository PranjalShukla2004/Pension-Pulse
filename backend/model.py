import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.graph_objects as go

# Hyperparameters and constants
NUM_BANKS = 3
NUM_CLIENTS = 1000
DEPOSIT_PER_CLIENT = 1000  # €1000 per client
EURIBOR_MEAN = 0.01  # 1% baseline
EURIBOR_VOLATILITY = 0.002
BANK_MARGIN = 0.015  # Bank's return = EURIBOR + margin
EPISODES = 50
STEPS_PER_EPISODE = 24  # Simulating 2 years monthly
MIN_RATE = 0.0
MAX_RATE = 0.05  # 0-5% interest rates
CLIENT_ATTRIBUTES = 3  # wealth, risk_tolerance, loyalty
ECON_FACTORS = 2       # inflation, gdp_growth
STATE_SIZE = 6  # 3 client attributes + 2 economic factors + 1 EURIBOR value
LEARNING_RATE = 0.001

# Advanced Bank Agent using policy gradient
class AdvancedBankAgent:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.std_dev = 0.005  # Standard deviation for sampling actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(STATE_SIZE,)),
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
        mean_rate = self.model.predict(scaled_state, verbose=0)[0][0] * MAX_RATE
        # Sample an action from a Gaussian distribution centered at mean_rate
        action = np.random.normal(mean_rate, self.std_dev)
        return np.clip(action, MIN_RATE, MAX_RATE)
    
    def predict_rate(self, state):
        """Return the deterministic predicted rate without noise."""
        scaled_state = self.scaler.transform([state])
        predicted = self.model.predict(scaled_state, verbose=0)[0][0] * MAX_RATE
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
            predicted_mean = predicted * MAX_RATE
            # Compute the log probability under a Gaussian with fixed std_dev
            log_probs = -0.5 * tf.square((actions - predicted_mean) / self.std_dev) \
                        - tf.math.log(self.std_dev * tf.sqrt(2 * np.pi))
            # Loss is negative log likelihood weighted by reward (policy gradient objective)
            loss = -tf.reduce_mean(log_probs * rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


# Enhanced Banking Environment
class EnhancedBankEnvironment:
    def __init__(self):
        self.client_attributes = np.zeros((NUM_CLIENTS, CLIENT_ATTRIBUTES))
        self.econ_factors = np.zeros(ECON_FACTORS)
        
    def reset(self):
        # Generate synthetic clients
        self.client_attributes = np.column_stack((
            np.random.lognormal(5, 0.5, NUM_CLIENTS),  # Wealth
            np.random.beta(2, 5, NUM_CLIENTS),          # Risk tolerance
            np.random.uniform(0, 1, NUM_CLIENTS)        # Loyalty
        ))
        # Generate economic factors
        self.econ_factors = np.array([
            np.clip(np.random.normal(0.02, 0.005), 0, 0.1),  # Inflation
            np.random.normal(0.03, 0.01)                     # GDP growth
        ])
        return self._get_state()
    
    def _get_state(self):
        avg_client_attributes = np.mean(self.client_attributes, axis=0)  # Shape: (3,)
        econ_factors = self.econ_factors  # Shape: (2,)
        euribor = np.array([np.random.normal(EURIBOR_MEAN, EURIBOR_VOLATILITY)])  # Shape: (1,)
        # State is a 6D vector: [avg_client_attributes, econ_factors, euribor]
        state = np.concatenate((avg_client_attributes, econ_factors, euribor))
        return state
    
    def calculate_utility(self, rate, agent_id):
        wealth = self.client_attributes[:, 0]
        risk_tol = self.client_attributes[:, 1]
        loyalty = self.client_attributes[:, 2]
        return (rate * 100) + (wealth / 1000) - (risk_tol * 5) + (loyalty * agent_id * 2)


def train_agents(episodes=EPISODES):
    env = EnhancedBankEnvironment()
    agent = AdvancedBankAgent()
    
    # Pretrain the scaler using synthetic scenarios
    synthetic_scenarios = []
    for _ in tqdm(range(10000), desc="Fitting scaler"):
        scenario = np.concatenate((
            np.random.lognormal(5, 0.5, CLIENT_ATTRIBUTES),
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
        
        for step in range(STEPS_PER_EPISODE):
            # Agent selects an interest rate based on the current state
            rate = agent.act(state)
            episode_rates.append(rate)
            
            # Simulate client acquisition
            utilities = env.calculate_utility(rate, 1)  # Assume agent_id = 1
            clients_acquired = np.sum(utilities > 0)  # Count clients with positive utility
            
            # Calculate profit based on the rate chosen and client acquisition
            profit = (EURIBOR_MEAN + BANK_MARGIN - rate) * clients_acquired * DEPOSIT_PER_CLIENT
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
            agent.model.save(f'advanced_bank_agent_episode_{episode+1}.h5')
    
    # Save the final model
    agent.model.save('advanced_bank_agent_final.h5')
    return agent, history


def plot_2d_relationships(agent):
    """
    For each input dimension, vary that input while holding others constant,
    and plot the predicted interest rate (output).
    """
    # Default state: [wealth, risk_tolerance, loyalty, inflation, gdp_growth, euribor]
    defaults = [500, 0.5, 0.5, 0.02, 0.03, 0.01]
    # Define ranges for each input feature
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
    
    for i in range(STATE_SIZE):
        values = ranges[i]
        predictions = []
        for v in values:
            state = defaults.copy()
            state[i] = v
            # Use the deterministic prediction method
            pred = agent.predict_rate(state)
            predictions.append(pred)
        
        plt.figure(figsize=(8, 5))
        plt.plot(values, predictions, color='blue', linewidth=2)
        plt.xlabel(feature_names[i])
        plt.ylabel('Predicted Interest Rate')
        plt.title(f'2D Relationship: {feature_names[i]} vs. Interest Rate')
        plt.grid(True)
        plt.savefig(f'2d_plot_{feature_names[i].lower().replace(" ", "_")}.png')
        plt.show()


def plot_3d_interactive(agent):
    """
    For each pair of input features, create a 3D interactive plot (using Plotly)
    with the two features on the x and y axes and the predicted interest rate as z.
    Other features are held constant at default values.
    """
    defaults = [500, 0.5, 0.5, 0.02, 0.03, 0.01]
    feature_names = {
        0: ('Wealth', np.linspace(50, 2000, 50)),
        1: ('Risk Tolerance', np.linspace(0, 1, 50)),
        2: ('Loyalty', np.linspace(0, 1, 50)),
        3: ('Inflation', np.linspace(0, 0.1, 50)),
        4: ('GDP Growth', np.linspace(0, 0.1, 50)),
        5: ('EURIBOR', np.linspace(0, 0.05, 50))
    }
    
    # Loop through each pair (i, j)
    for i in range(STATE_SIZE):
        for j in range(i+1, STATE_SIZE):
            name_i, range_i = feature_names[i]
            name_j, range_j = feature_names[j]
            X, Y = np.meshgrid(range_i, range_j)
            Z = np.zeros_like(X)
            
            for idx in range(X.shape[0]):
                for jdx in range(X.shape[1]):
                    state = defaults.copy()
                    state[i] = X[idx, jdx]
                    state[j] = Y[idx, jdx]
                    Z[idx, jdx] = agent.predict_rate(state)
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
            fig.update_layout(
                title=f'3D Surface: {name_i} vs. {name_j} vs. Interest Rate',
                scene = dict(
                    xaxis_title=name_i,
                    yaxis_title=name_j,
                    zaxis_title='Interest Rate'
                )
            )
            filename = f'3d_plot_{name_i.lower().replace(" ", "_")}_{name_j.lower().replace(" ", "_")}.html'
            fig.write_html(filename)
            print(f"Saved interactive plot: {filename}")


# Example usage:
if __name__ == '__main__':
    # Train the agent and collect training history
    agent, history = train_agents(episodes=EPISODES)
    
    # Analyze and plot strategy (existing 2D plot example from wealth vs. rate)
    def analyze_strategy(agent):
        # Generate a grid of synthetic states over wealth
        wealth_values = np.linspace(50, 2000, 100)
        results = []
        risk = 0.5  # Fixed risk tolerance
        for wealth in tqdm(wealth_values, desc="Analyzing strategy"):
            # wealth, risk, loyalty, inflation, gdp_growth, euribor
            synthetic_state = [wealth, risk, 0.5, 0.02, 0.03, 0.01]
            rate = agent.predict_rate(synthetic_state)
            results.append((wealth, rate))
        
        plt.figure(figsize=(12, 6))
        plt.plot([x for x, _ in results],
                 [y for _, y in results],
                 color='red', linewidth=2)
        plt.axhline(y=EURIBOR_MEAN, color='black', linestyle='--', label='EURIBOR')
        plt.xlabel('Average Client Wealth')
        plt.ylabel('Interest Rate')
        plt.title('Bank Strategy Optimization: Wealth vs. Interest Rate')
        plt.legend()
        plt.savefig('bank_agent_strategy.png')
        plt.show()
    
    analyze_strategy(agent)
    
    # Generate 2D plots for each input feature
    plot_2d_relationships(agent)
    
    # Generate interactive 3D plots for each pair of input features
    plot_3d_interactive(agent)
