import Environment as Env
import Agents
import random


N_AGENTS = 2
EPISODES = 1_000_000
ALPHA = 0.03
GAMMA = 0.99
EPSILON_START = 0
NUM_COINS = 3
filename_save = 'Agents/Agent_1/Models/model'
model_load = 'model_1000000.npy'

TRAIN = False

if __name__ == "__main__":
    env = Env.CoinCollectionEnv(grid_size=10, num_agents=N_AGENTS, num_coins=NUM_COINS, max_steps=100)
    agents = Agents.MultiAgent(n_agents=N_AGENTS, env=env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon_start=EPSILON_START)
    if TRAIN:
        agents.train()
        env.reset(NUM_COINS)
    else:
        agents.load_models(model_load)

    for e in range(100):
        done = False
        while not done:
            actions = agents.choose_actions(state=env.get_obs())
            obs, rewards, done = env.step(actions)
            print('Obs:', obs, "Actions:", actions, "Rewards:", rewards)
            env.render()

        env.reset(random.randint(1, 5))

    env.out.release()