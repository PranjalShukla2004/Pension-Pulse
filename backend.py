from agent import BankAgent, BankEnvironment
import numpy as np
from tqdm import tqdm
import config


def train_agents(episodes=config.EPISODES):
    env = BankEnvironment()
    # Create one agent per bank
    agents = [BankAgent() for _ in range(config.NUM_BANKS)]
    
    # Pretrain each agent's scaler using synthetic scenarios
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
    
    # Initialise training history for each bank
    history = {i: {'rates': [], 'profits': []} for i in range(config.NUM_BANKS)}
    
    for episode in tqdm(range(episodes), desc="Training episodes"):
        state = env.reset()
        
        # To record episode data per bank
        episode_rates = [ [] for _ in range(config.NUM_BANKS) ]
        episode_profits = [ [] for _ in range(config.NUM_BANKS) ]
        
        for step in range(config.STEPS_PER_EPISODE):
            # Each bank offers its own rate given the same state
            rates = [agent.act(state) for agent in agents]
            
            # For each bank, compute the per-customer utility
            utilities = []
            for bank_id, rate in enumerate(rates):
                util = env.calculate_utility(rate, bank_id+1)  # bank_id+1 used as a proxy for reputation
                utilities.append(util)
            # Convert to a (NUM_BANKS, NUM_CLIENTS) array
            utilities = np.array(utilities)
            # For each customer, choose the bank that provides the maximum utility
            chosen_bank_indices = np.argmax(utilities, axis=0)
            
            # Count clients acquired per bank
            clients_acquired = [np.sum(chosen_bank_indices == i) for i in range(config.NUM_BANKS)]
            
            # Calculate profits for each bank
            profits = []
            for i in range(config.NUM_BANKS):
                profit = (config.EURIBOR_MEAN + config.BANK_MARGIN - rates[i]) * clients_acquired[i] * config.DEPOSIT_PER_CLIENT
                profits.append(profit)
            
            # Train each bank's agent with its own experience (state, offered rate, profit)
            for i, agent in enumerate(agents):
                agent.train([state], [rates[i]], [profits[i]])
                episode_rates[i].append(rates[i])
                episode_profits[i].append(profits[i])
            
            # Update state for the next step
            state = env._get_state()
        
        # Record average rate and profit per bank for this episode
        for i in range(config.NUM_BANKS):
            avg_rate = np.mean(episode_rates[i])
            avg_profit = np.mean(episode_profits[i])
            history[i]['rates'].append(avg_rate)
            history[i]['profits'].append(avg_profit)
        
        if (episode + 1) % 100 == 0:
            info = " | ".join(
                [f"Bank {i+1}: Avg Rate {np.mean(episode_rates[i]):.4f}, Avg Profit {np.mean(episode_profits[i]):.2f}" 
                 for i in range(config.NUM_BANKS)]
            )
            print(f"Episode {episode + 1}/{episodes} | {info}")
            # Save models periodically
            for i, agent in enumerate(agents):
                agent.model.save(f'advanced_bank_agent_bank_{i+1}_episode_{episode+1}.h5')
    
    # Save final models
    for i, agent in enumerate(agents):
        agent.model.save(f'advanced_bank_agent_bank_{i+1}_final.h5')
    
    return agents, history

