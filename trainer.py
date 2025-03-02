from models import train_agents
import config


agents, _ = train_agents(episodes=config.EPISODES)
for i, agent in enumerate(agents):
    db_path = config.PREDICTIONS_DB_PATH_TEMPLATE.format(bank_id=i+1)
