import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, Sequential, Input, losses
from tensorflow.keras.models import load_model
from typing import Optional
import random
import os
import pickle
import copy


def build_agent_obs(state, agent_idx, n_agents):
    xs, ys, mask = state
    own_x, own_y = xs[agent_idx], ys[agent_idx]
    rel_pos = [coord for i in range(n_agents) if i != agent_idx for coord in (xs[i] - own_x, ys[i] - own_y)]

    return np.concatenate([[own_x, own_y], rel_pos, mask]).astype(np.float32)

def build_agent_net(obs_dim: int, n_actions: int, hidden_sizes=(128, 128)) -> tf.keras.Model:
    return Sequential(
        [Input(shape=(obs_dim,))] +
        [layers.Dense(units=h, activation='relu') for h in hidden_sizes] +
        [layers.Dense(units=n_actions, activation='linear')]
    )

def update_epsilon(epsilon_start, epsilon_end, progress):
    return max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * progress)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(buf[idx] for buf in (self.obs, self.actions, self.rewards, self.next_obs, self.dones))

    def __len__(self):
        return self.size

class SingleAgentDQN:
    def __init__(self, env, obs_dim, n_actions, **cfg):
        self.env = env
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.gamma = cfg.get("gamma", 0.99)
        self.batch_size = cfg.get("batch_size", 64)
        self.target_update_freq = cfg.get("target_update_freq", 1000)

        self.epsilon_start = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.05)
        self.epsilon = self.epsilon_start

        self.model = build_agent_net(obs_dim, n_actions, cfg.get("hidden_sizes", (128, 128)))
        self.target_model = build_agent_net(obs_dim, n_actions, cfg.get("hidden_sizes", (128, 128)))
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = optimizers.Adam(cfg.get("lr", 1e-4))
        self.loss_fn = losses.Huber()
        self.replay = ReplayBuffer(cfg.get("buffer_capacity", 50_000), obs_dim)

        self.train_steps = 0

    def choose_action(self, obs, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q_vals = self.model(obs[None], training=False)[0]
        return int(tf.argmax(q_vals))

    @tf.function
    def train_step(self, states_t, actions_t, rewards_t, next_states_t, dones_t):
        with tf.GradientTape() as tape:
            # Current Q-values
            q_vals = self.model(states_t, training=True)  # (B, n_actions)

            # Q(s', ·) from online network for action selection
            q_next_online = self.model(next_states_t, training=False)  # (B, n_actions)
            next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)  # (B,)

            # Q(s', ·) from target network for evaluation
            q_next_target = self.target_model(next_states_t, training=False)  # (B, n_actions)

            # gather target Q-values for chosen actions
            batch_range = tf.range(tf.shape(next_states_t)[0], dtype=tf.int32)
            indices = tf.stack([batch_range, next_actions], axis=1)  # (B, 2)
            q_next_chosen = tf.gather_nd(q_next_target, indices)  # (B,)

            targets = rewards_t + (1.0 - dones_t) * self.gamma * q_next_chosen  # (B,)

            # gather predicted Q-values for taken actions
            indices_taken = tf.stack([batch_range, actions_t], axis=1)
            q_taken = tf.gather_nd(q_vals, indices_taken)  # (B,)

            loss = self.loss_fn(targets, q_taken)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def train_from_replay(self, n_updates=1):
        if len(self.replay) < self.batch_size:
            return None

        losses = []
        for _ in range(n_updates):
            s_b, a_b, r_b, ns_b, d_b = self.replay.sample(self.batch_size)
            # convert to tensors
            s_t = tf.convert_to_tensor(s_b, dtype=tf.float32)
            a_t = tf.convert_to_tensor(a_b, dtype=tf.int32)
            r_t = tf.convert_to_tensor(r_b, dtype=tf.float32)
            ns_t = tf.convert_to_tensor(ns_b, dtype=tf.float32)
            d_t = tf.convert_to_tensor(d_b, dtype=tf.float32)

            loss = self.train_step(s_t, a_t, r_t, ns_t, d_t)
            losses.append(float(loss.numpy()))

            self.train_step_count += 1
            if self.train_step_count % self.target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

        return float(np.mean(losses)) if losses else None

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath)
        self.target_model.set_weights(self.model.get_weights())


class MultiAgentIQL_DQN:
    def __init__(self, n_agents, env, obs_dim, **kwargs):
        self.n_agents = n_agents
        self.env = env
        self.agents = [SingleAgentDQN(env, obs_dim, len(env.action_space), **kwargs) for _ in range(n_agents)]

    def choose_actions(self, state):
        return [agent.choose_action(build_agent_obs(state, i, self.n_agents)) for i, agent in enumerate(self.agents)]

    def train(self, episodes, save=False, n_states_to_save: int = 0, updates_per_step=4):
        returns = []
        saved_init_states = []
        steps_list = []
        env_rewards_list = []
        for e in range(episodes):
            state = self.env.reset(random.randint(0, 5))
            if n_states_to_save and e >= (episodes - n_states_to_save):
                saved_init_states.append(copy.deepcopy(state))

            done = False
            steps = 0
            ep_rewards = [0] * self.n_agents
            env_reward = 0
            while not done:
                actions = self.choose_actions(state)
                next_state, rewards, done = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    obs = build_agent_obs(state, i, self.n_agents)
                    next_obs = build_agent_obs(next_state, i, self.n_agents)
                    agent.replay.add(obs, actions[i], rewards[i], next_obs, done)
                    ep_rewards[i] += rewards[i]
                    if rewards[i] > 0:
                        env_reward += rewards[i]
                    _ = agent.train_from_replay(n_updates=updates_per_step)

                state = next_state
                steps += 1
                env_reward -= 3

            for agent in self.agents:
                agent.epsilon = update_epsilon(agent.epsilon_start, agent.epsilon_end, e / (0.6 * episodes))
            steps_list.append(steps)
            env_rewards_list.append(env_reward)
            returns.append(ep_rewards)
            if e % max(1, (episodes // 10)) == 0:
                print(f"Episode {e} mean returns: {np.mean(returns, axis=0)}, epsilon: {self.agents[0].epsilon}")
                returns.clear()
                if save:
                    self.save_models(f"model_{e}")

        iql_results = {
            'rewards': env_rewards_list,
            'steps': steps_list
        }
        with open('Plots/training_info/iql_results.pkl', 'wb') as f:
            pickle.dump(iql_results, f)
        if save:
            self.save_models(f"model_{episodes}")
        return saved_init_states

    def save_models(self, model_name):
        for i, agent in enumerate(self.agents):
            fn = f"Agents/IQL/Agent_{i+1}/Models/{model_name}.keras"
            agent.save(fn)

    def load_models(self, model_name):
        for i, agent in enumerate(self.agents):
            fn = f"Agents/IQL/Agent_{i+1}/Models/{model_name}"
            agent.load(fn)


class VDNTrainer:
    def __init__(
            self,
            n_agents: int,
            obs_dim: int,
            n_actions: int,
            lr: float = 5e-4,
            gamma: float = 0.99,
            buffer_capacity: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 1000,
            hidden_sizes=(128, 128),
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)

        # Networks
        self.agent_net = build_agent_net(obs_dim, n_actions, hidden_sizes)
        self.target_agent_net = build_agent_net(obs_dim, n_actions, hidden_sizes)
        self.target_agent_net.set_weights(self.agent_net.get_weights())
        self.agent_nets = [self.agent_net] * n_agents
        self.target_agent_nets = [self.target_agent_net] * n_agents

        # Optimizer & loss
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.train_step = 0

    def choose_actions(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if random.random() < epsilon:
            return np.random.randint(self.n_actions, size=self.n_agents)

        obs = np.asarray(obs, dtype=np.float32)
        q_vals = self.agent_net(obs, training=False).numpy()
        return np.argmax(q_vals, axis=1).astype(np.int32)

    @tf.function
    def _train_on_batch(self, obs, actions, rewards, next_obs, dones):
        B = tf.shape(obs)[0]
        with tf.GradientTape() as tape:
            obs_flat = tf.reshape(obs, (B * self.n_agents, self.obs_dim))
            next_obs_flat = tf.reshape(next_obs, (B * self.n_agents, self.obs_dim))

            q = self.agent_net(obs_flat, training=True)
            q = tf.reshape(q, (B, self.n_agents, self.n_actions))

            q_next_online = self.agent_net(next_obs_flat, training=False)
            q_next_online = tf.reshape(q_next_online, (B, self.n_agents, self.n_actions))
            next_actions = tf.argmax(q_next_online, axis=-1, output_type=tf.int32)

            q_next_target = self.target_agent_net(next_obs_flat, training=False)
            q_next_target = tf.reshape(q_next_target, (B, self.n_agents, self.n_actions))

            batch_idx = tf.range(B)[:, None]
            #batch_idx_tiled = tf.tile(batch_idx, [1, self.n_agents])
            agent_idx = tf.range(self.n_agents)[None, :]
            #agent_idx_tiled = tf.tile(agent_idx, [B, 1])

            idx = tf.stack([batch_idx, agent_idx, actions], axis=-1)
            idx_next = tf.stack([batch_idx, agent_idx, next_actions], axis=-1)

            q_taken = tf.gather_nd(q, idx)
            q_next = tf.gather_nd(q_next_target, idx_next)

            q_team = tf.reduce_sum(q_taken, axis=1)
            q_next_team = tf.reduce_sum(q_next, axis=1)

            targets = rewards + (1.0 - dones) * self.gamma * q_next_team

            loss = self.mse(targets, q_team)

        grads = tape.gradient(loss, self.agent_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.agent_net.trainable_variables))
        return loss

    def update_from_buffer(self, n_updates: int = 1) -> Optional[float]:
        if self.buffer.size < self.batch_size:
            return None

        losses = []
        for _ in range(n_updates):
            obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.buffer.sample(self.batch_size)
            # convert to tensors
            obs_t = tf.convert_to_tensor(obs_b, dtype=tf.float32)
            actions_t = tf.convert_to_tensor(actions_b, dtype=tf.int32)
            rewards_t = tf.convert_to_tensor(rewards_b, dtype=tf.float32)
            next_obs_t = tf.convert_to_tensor(next_obs_b, dtype=tf.float32)
            dones_t = tf.convert_to_tensor(dones_b, dtype=tf.float32)

            loss = self._train_on_batch(obs_t, actions_t, rewards_t, next_obs_t, dones_t)
            losses.append(float(loss.numpy()))
            self.train_step += 1

            if self.train_step % self.target_update_freq == 0:
                self.update_target_networks()

        return float(np.mean(losses)) if losses else None

    def run_episode(self, env, epsilon, init_state=None):
        if init_state is None:
            state = env.reset(random.randint(1, 5))
        else:
            state = init_state
            agents_pos = [[state[0][0], state[1][0]], [state[0][1], state[1][1]]]
            coins = env.get_coins_from_mask(state[2])
            env.set_init_state(agents_pos, coins)

        obs = [self.get_obs(state, agent_idx=i) for i in range(self.n_agents)]
        env_return = 0
        steps = 0
        team_return = 0
        done = False
        while not done:
            actions = self.choose_actions(obs, epsilon)
            next_state, rewards, done = env.step(actions)
            next_obs = [self.get_obs(next_state, agent_idx=i) for i in range(self.n_agents)]
            team_reward = sum(rewards)
            team_return += team_reward
            self.buffer.add(obs, actions, team_reward, next_obs, done)
            _ = self.update_from_buffer(2)
            obs = next_obs
            for r in rewards:
                if r > 0:
                    env_return += r
            env_return -= 3
            steps += 1

        return env_return, team_return, steps

    def train(self, env, episodes, epsilon_start, epsilon_end, save: bool = False, init_states: list = None):
        env_rewards, team_rewards, steps = [], [], []
        for e in range(episodes):
            epsilon = update_epsilon(epsilon_start, epsilon_end, e / (0.6 * episodes))
            init_state = None if init_states is None else init_states[e % len(init_states)]
            env_reward, team_return, ep_steps = self.run_episode(env, epsilon, init_state)

            env_rewards.append(env_reward)
            steps.append(ep_steps)
            team_rewards.append(team_return)
            if e % (episodes // 10) == 0:
                print(f'Episode {e}, Expected Returns: {np.mean(team_rewards, axis=0)}, Epsilon: {epsilon:.3f}')
                team_rewards.clear()
                if save:
                    self.save(f'model_{e}')

        vdn_results = {
            'rewards': env_rewards,
            'steps': steps
        }
        with open('Plots/training_info/vdn_results.pkl', 'wb') as f:
            pickle.dump(vdn_results, f)
        if save:
            self.save(f'model_{episodes}')

        env.reset(random.randint(0, 5))
        print('Training has completed!')

    def update_target_networks(self):
        self.target_agent_net.set_weights(self.agent_net.get_weights())

    def save(self, prefix: str):
        self.agent_net.save(f"Agents/VDN/Shared/agent_net/{prefix}.keras")
        self.target_agent_net.save(f"Agents/VDN/Shared/target/{prefix}.keras")

    def load(self, model: str):
        self.agent_net = load_model(f"Agents/VDN/Shared/agent_net/{model}.keras")
        self.target_agent_net = load_model(f"Agents/VDN/Shared/target/{model}.keras")

    def get_obs(self, state, agent_idx=0):
        return build_agent_obs(state, agent_idx, self.n_agents)
