import numpy as np
import random


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

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(len(self.env.action_space)))
        else:
            state = self._get_state_values(state)
            return np.argmax(self.q_table[state])

    def _get_mask_idx(self, mask):
        binary_str_from_mask = ''.join(map(str, mask))
        return int(binary_str_from_mask, 2)

    def _get_state_values(self, state):
        x, y = int(state[0][0]), int(state[1][0])
        mask = state[2]
        mask_idx = self._get_mask_idx(mask)
        return x, y, mask_idx

    def update_q_table(self, state, next_state, action, reward):
        state = self._get_state_values(state)
        #print('State:', state)
        next_state = self._get_state_values(next_state)
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

    def load_model(self, filename: str):
        self.q_table = np.load(filename)
