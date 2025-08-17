import numpy as np
import random


class SingleAgent:
    def __init__(self, env, alpha, epsilon_start, epsilon_end, epsilon_decay, gamma, episodes):
        self.env = env

        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.episodes = episodes
        '''
        self.grid_size = grid_size
        '''

        self.q_table = np.zeros(shape=(env.grid_size, env.grid_size, 2 ** len(env.coin_slots), len(env.action_space)))

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(len(self.env.action_space)))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, next_state, action, reward):
        self.q_table[state, action] += self.alpha * (reward * self.gamma * self.q_table[next_state, action] - self.q_table[state, action])

    def train(self):
        epsilon = self.epsilon_start

        for e in range(self.episodes):
            state = self.env.get_obs()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, next_state, action, reward)


