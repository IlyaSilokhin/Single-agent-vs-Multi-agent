import Environment as Env
import Agents
import plots
import pickle


def print_results(results):
    for algo, stats in results.items():
        print(f"\nAlgorithm: {algo}")
        print(f"  Average Episodic Reward: {stats['avg_reward']:.2f}")
        print(f"  Average Episode Length: {stats['avg_length']:.2f} steps")
        print("  Coin Collection per Agent:")
        for i, pct in enumerate(stats["per_agent_coin_percent"]):
            print(f"    Agent {i}: {pct*100:.1f}% of coins")

N_AGENTS = 2
EPISODES = 15000
ALPHA = 0.0003
GAMMA = 0.99
EPSILON_START = 0
N_COINS = 3
filename_save = 'Agents/Agent_test/Models/model'
model_load = 'model_15000.keras'
plots_prefix = 'Plots/plot_4a'

TRAIN = False

if __name__ == "__main__":
    env_iql = Env.CoinCollectionEnv(grid_size=8, num_agents=N_AGENTS, num_coins=N_COINS, max_steps=50)
    env_vdn = Env.CoinCollectionEnv(grid_size=8, num_agents=N_AGENTS, num_coins=N_COINS, max_steps=50)
    obs_dim = 5 + 2 * N_AGENTS
    agents_iql = Agents.MultiAgentIQL_DQN(n_agents=N_AGENTS, env=env_iql, obs_dim=obs_dim, lr=ALPHA, gamma=GAMMA, epsilon_start=EPSILON_START)
    agents_vdn = Agents.VDNTrainer(n_agents=N_AGENTS, obs_dim=obs_dim, n_actions=4, lr=ALPHA, gamma=GAMMA, batch_size=64, target_update_freq=200)
    if TRAIN:
        init_states = agents_iql.train(episodes=EPISODES, save=True, n_states_to_save=EPISODES)
        agents_vdn.train(env=env_vdn, episodes=EPISODES, epsilon_start=EPSILON_START, epsilon_end=0.1, save=True, init_states=init_states)
        with open('Plots/training_info/iql_results.pkl', 'rb') as f:
            iql_results = pickle.load(f)
        with open('Plots/training_info/vdn_results.pkl', 'rb') as f:
            vdn_results = pickle.load(f)

        plots.plot_training(first_results=iql_results, second_results=vdn_results, save_prefix='evaluation', first_prefix='Multi-Agent_IQL', second_prefix='Multi-Agent_VDN')
    else:
        agents_iql.load_models(model_load)
        agents_vdn.load(model=f'model_{EPISODES}')

    stats = Env.run_split_screen(N_AGENTS, N_COINS, env_iql, env_vdn, agents_iql, agents_vdn, episodes=100)
    print_results(stats)
