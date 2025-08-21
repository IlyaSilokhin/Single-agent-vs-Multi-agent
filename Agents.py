import numpy as np
import random
import os


class SingleAgent:
    def __init__(self, env, episodes, alpha=0.01, epsilon_start=1.0, epsilon_end=0, gamma=0.95):
        self.env = env

        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_start / (episodes / 2)
        self.gamma = gamma
        self.episodes = episodes

        self.epsilon = epsilon_start
        self.q_table = np.zeros(shape=(env.grid_size, env.grid_size, 2 ** len(env.coin_slots), len(env.action_space)))

    def choose_action(self, state, agent_idx = 0):
        if random.random() < self.epsilon:
            return random.choice(range(len(self.env.action_space)))
        else:
            state = self.get_state_values(state, agent_idx)
            return np.argmax(self.q_table[state])

    def _get_mask_idx(self, mask):
        binary_str_from_mask = ''.join(map(str, mask))
        return int(binary_str_from_mask, 2)

    def get_state_values(self, state, agent_idx = 0):
        x, y = int(state[0][agent_idx]), int(state[1][agent_idx])
        mask = state[2]
        mask_idx = self._get_mask_idx(mask)
        return x, y, mask_idx

    def update_q_table(self, state, next_state, action, reward, agent_idx = 0):
        state = self.get_state_values(state, agent_idx)
        #print('State:', state)
        next_state = self.get_state_values(next_state, agent_idx)
        self.q_table[state[0], state[1], state[2], action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], next_state[2]]) - self.q_table[state[0], state[1], state[2], action])

    def train(self, filename: str):
        returns = []
        for e in range(self.episodes):
            state = self.env.reset(random.randint(0, 5))
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, rewards, done = self.env.step([action])
                reward = rewards[0]
                self.update_q_table(state, next_state, action, reward)
                total_reward += reward
                state = next_state

            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_end)
            returns.append(total_reward)
            if e % (self.episodes // 10) == 0:
                print(f'Episode: {e}, Expected Return: {np.mean(returns)}')
                self.save_model(filename + '_' + str(e) + '.npy')
                returns.clear()

        self.save_model(filename + '_' + str(self.episodes) + '.npy')
        print('Agent has successfully trained!')

    def save_model(self, filename: str):
        np.save(filename, self.q_table)

    def load_model(self, model: str, idx = 0):
        filename = f'Agents/Agent_{idx + 1}/Models/{model}'
        self.q_table = np.load(filename)


class MultiAgent:
    def __init__(self, n_agents, env, episodes, alpha=0.01, epsilon_start=1.0, epsilon_end=0, gamma=0.95):
        self.n_agents = n_agents
        self.env = env

        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_start / (episodes / 2)
        self.gamma = gamma
        self.episodes = episodes

        self.agents = self.init_agents()

    def init_agents(self) -> tuple:
        agents = []
        for _ in range(self.n_agents):
            agent = SingleAgent(self.env, self.episodes, self.alpha, self.epsilon_start, self.epsilon_end, self.gamma)
            agents.append(agent)
        return tuple(agents)

    def choose_actions(self, state) -> list:
        return [agent.choose_action(state, agent_idx=i) for i, agent in enumerate(self.agents)]

    def get_i_agent_state(self, state, idx):
        return self.agents[idx].get_state_values(state, agent_idx=idx)

    def train(self):
        returns = []
        for e in range(self.episodes):
            state = self.env.reset(random.randint(0, 5))
            total_rewards = [0] * self.n_agents
            done = False
            while not done:
                actions = self.choose_actions(state)
                next_state, rewards, done = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    agent.update_q_table(state, next_state, actions[i], rewards[i], agent_idx=i)
                    total_rewards[i] += rewards[i]
                state = next_state

            returns.append(total_rewards)
            for i, agent in enumerate(self.agents):
                agent.epsilon = max(agent.epsilon - self.epsilon_decay, self.epsilon_end)
            if e % (self.episodes // 10) == 0:
                print(f'Episode {e}, Expected Returns: {np.mean(returns, axis=0)}, Epsilon_0: {self.agents[0].epsilon}')
                returns.clear()
                self.save_models(f'model_{e}')

        self.save_models(f'model_{self.episodes}')
        print('Agents have successfully trained!')

    def save_models(self, model_name: str):
        for i, agent in enumerate(self.agents):
            os.makedirs(f'Agents/Agent_{i + 1}/Models', exist_ok=True)
            filename = f'Agents/Agent_{i + 1}/Models/{model_name}.npy'
            agent.save_model(filename)

    def load_models(self, model: str):
        for i, agent in enumerate(self.agents):
            agent.load_model(model, idx = i)

