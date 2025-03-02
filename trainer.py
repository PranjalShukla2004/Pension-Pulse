from models import train_agents, save_model_predictions_to_db
import config


agents, _ = train_agents(episodes=config.EPISODES)
for i, agent in enumerate(agents):
    db_path = config.PREDICTIONS_DB_PATH_TEMPLATE.format(bank_id=i+1)
    save_model_predictions_to_db(agent, db_path=db_path)
