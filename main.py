import Environment as Env
import Agents
import random


EPISODES = 1_000_000
ALPHA = 0.03
GAMMA = 0.99
EPSILON_START = 0
NUM_COINS = 3
filename_save = 'Models/model'
filename_load = 'Models/model_1000000.npy'

TRAIN = False

if __name__ == "__main__":
    env = Env.CoinCollectionEnv(grid_size=10, num_agents=1, num_coins=NUM_COINS, max_steps=100)
    agent = Agents.SingleAgent(env=env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon_start=EPSILON_START)
    if TRAIN:
        agent.train(filename_save)
        env.reset(NUM_COINS)
    else:
        agent.load_model(filename_load)

    for e in range(100):
        done = False
        while not done:
            actions = [agent.choose_action(state=env.get_obs())]
            obs, rewards, done = env.step(actions)
            print('Obs:', obs, "Actions:", actions, "Rewards:", rewards)
            env.render()

        env.reset(random.randint(1, 5))