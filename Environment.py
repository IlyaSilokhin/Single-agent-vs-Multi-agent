import numpy as np
import random
import pygame
import cv2
from collections import defaultdict


MOVE_REWARD = -3
COIN_REWARD = 10
FPS = 4
MAX_COINS = 5
MAX_AGENTS = 4


class CoinCollectionEnv:
    def __init__(self, grid_size=5, num_agents=1, num_coins=3, max_steps=50, cell_size=80, record=True):
        self._rng = random.Random()
        self.seed(0)

        self.grid_size = grid_size
        self.num_agents = min(num_agents, MAX_AGENTS)
        self.num_coins = min(num_coins, MAX_COINS)
        self.max_steps = max_steps
        self.cell_size = cell_size
        self.record = record

        self.coin_slots = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1), ((grid_size - 1) // 2, (grid_size - 1) // 2)]
        self.visitation_map = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.action_space = [0, 1, 2, 3, 4]
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }

        self.agent_positions = []
        self.coins = []
        self.steps = 0

        self._init_rendering()
        self.reset(num_coins)

    def seed(self, seed_value):
        self._rng.seed(seed_value)
        np.random.seed(seed_value)

    def _init_rendering(self):
        pygame.init()
        size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Coin Collection Environment")
        self.clock = pygame.time.Clock()

        if self.record:
            self.out = cv2.VideoWriter('Records/game.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (size, size))

        self.bg_color = (255, 255, 255)
        self.grid_color = (200, 200, 200)
        self.coin_color = (255, 215, 0)
        self.agent_colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0), (255, 0, 255)]  # Support up to 4 agents

    def reset(self, n_coins):
        self.visitation_map.fill(0)
        self.num_coins = np.clip(n_coins, 1, MAX_COINS)
        self.agent_positions = [self._random_empty_agent_pos() for _ in range(self.num_agents)]
        self.coins = [self._random_coin_pos() for _ in range(self.num_coins)]
        self.steps = 0
        return self.get_obs()

    def step(self, actions):
        self.steps += 1
        self._move_agents(actions)
        rewards = self._compute_rewards()
        done = self._is_done()
        return self.get_obs(), rewards, done

    def _move_agents(self, actions):
        for i, action in enumerate(actions):
            dx, dy = self.action_map[action]
            x, y = self.agent_positions[i]
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)
            self.agent_positions[i] = (new_x, new_y)
            self.visitation_map[new_x, new_y] += 1

    def _compute_rewards(self):
        rewards = [MOVE_REWARD] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            if pos in self.coins:
                rewards[i] = COIN_REWARD
                self.coins.remove(pos)
        return rewards

    def _is_done(self):
        return len(self.coins) == 0 or self.steps >= self.max_steps

    def get_obs(self):
        xs, ys = zip(*self.agent_positions)
        return list(xs), list(ys), self._get_coins_mask()

    def _get_coins_mask(self):
        return [1 if slot in self.coins else 0 for slot in self.coin_slots]

    def render(self, surface=None):
        surface = surface or self.screen
        surface.fill(self.bg_color)

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                visits = self.visitation_map[x, y]
                color = (255, 255 // 2, 10) if visits > 0 else self.bg_color
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, self.grid_color, rect, 1)

        # Draw coins
        for (cx, cy) in self.coins:
            center = (
                cy * self.cell_size + self.cell_size // 2,
                cx * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(surface, self.coin_color, center, self.cell_size // 4)

        # Draw agents
        for i, (ax, ay) in enumerate(self.agent_positions):
            color = self.agent_colors[i % len(self.agent_colors)]
            rect = pygame.Rect(
                ay * self.cell_size + 5,
                ax * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )
            pygame.draw.rect(surface, color, rect)

        # Capture frame
        if surface is self.screen:
            self._capture_frame()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _capture_frame(self):
        if not self.record:
            return
        frame = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.out.write(frame)

    def _random_empty_agent_pos(self):
        while True:
            pos = (self._rng.randint(0, self.grid_size - 1), self._rng.randint(0, self.grid_size - 1))
            if pos not in self.agent_positions and pos not in self.coin_slots:
                return pos

    def _random_coin_pos(self):
        free = [pos for pos in self.coin_slots if pos not in self.coins]
        return self._rng.choice(free)

    def get_coins_from_mask(self, mask):
        return [slot for slot, bit in zip(self.coin_slots, mask) if bit]

    @staticmethod
    def get_agent_2_pos_from_relative(agent_1_pos, rel_pos):
        return rel_pos[0] + agent_1_pos[0], rel_pos[1] + agent_1_pos[1]

    def set_init_state(self, agent_positions, coins):
        self.visitation_map.fill(0)
        self.agent_positions = [tuple(p) for p in agent_positions]
        self.coins = [tuple(c) for c in coins]
        self.num_coins = len(self.coins)
        self.steps = 0
        return self.get_obs()


def sync_envs(env1, env2, n_coins, seed):
    env1.seed(seed)
    env2.seed(seed)
    state1 = env1.reset(n_coins)
    agent_positions = list(env1.agent_positions)
    coins = list(env1.coins)
    state2 = env2.set_init_state(agent_positions, coins)
    assert env1.agent_positions == env2.agent_positions
    assert env1.coins == env2.coins

    return state1, state2

def init_episode_stats(n_agents):
    return {
        "reward": 0,
        "steps": 0,
        "coins": {i: set() for i in range(n_agents)},
    }

def update_coin_stats(coin_dict, rewards, step):
    for agent_id, r in enumerate(rewards):
        if r > 0:
            coin_dict[agent_id].add(step)

def compute_coin_percentages(coin_dict):
    total = sum(len(v) for v in coin_dict.values())
    if total == 0:
        return {k: 0 for k in coin_dict}
    return {k: len(v) / total * 100 for k, v in coin_dict.items()}

def run_split_screen(n_agents, n_coins, env1, env2, agents_iql, agents_vdn, episodes=10):
    pygame.init()

    width = env1.grid_size * env1.cell_size + env2.grid_size * env2.cell_size
    height = max(env1.grid_size * env1.cell_size, env2.grid_size * env2.cell_size)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("IQL (left) vs VDN (right)")
    clock = pygame.time.Clock()

    rewards = {'IQL': [], 'VDN': []}
    lengths = {'IQL': [], 'VDN': []}
    coins = {'IQL': defaultdict(list), 'VDN': defaultdict(list)}

    for ep in range(episodes):
        state1, state2 = sync_envs(env1, env2, n_coins, random.randint(0, 10_000))

        obs_2 = [agents_vdn.get_obs(state2, agent_idx=i) for i in range(n_agents)]

        done1 = done2 = False
        stats_iql = init_episode_stats(n_agents)
        stats_vdn = init_episode_stats(n_agents)

        while not (done1 and done2):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            screen.fill((0, 0, 0))
            env1.render(screen.subsurface((0, 0, env1.grid_size * env1.cell_size, env1.grid_size * env1.cell_size)))
            env2.render(screen.subsurface((env1.grid_size * env1.cell_size, 0, env2.grid_size * env2.cell_size, env2.grid_size * env2.cell_size)))

            a1 = agents_iql.choose_actions(state1)
            a2 = agents_vdn.choose_actions(obs_2, epsilon=0)

            if not done1:
                state1, r1, done1 = env1.step(a1)
                stats_iql['reward'] += np.sum(r1)
                stats_iql['steps'] += 1
                update_coin_stats(stats_iql["coins"], r1, stats_iql["steps"])

            if not done2:
                state2, r2, done2 = env2.step(a2)
                obs_2 = [agents_vdn.get_obs(state2, agent_idx=i) for i in range(n_agents)]
                stats_vdn["reward"] += np.sum(r2)
                stats_vdn["steps"] += 1
                update_coin_stats(stats_vdn["coins"], r2, stats_vdn["steps"])

            separator_x = env1.grid_size * env1.cell_size
            pygame.draw.line(screen, (0, 0, 0), (separator_x, 0), (separator_x, height), 3)

            pygame.display.flip()
            clock.tick(3)

        rewards["IQL"].append(stats_iql["reward"])
        rewards["VDN"].append(stats_vdn["reward"])
        lengths["IQL"].append(stats_iql["steps"])
        lengths["VDN"].append(stats_vdn["steps"])

        for aid, pct in compute_coin_percentages(stats_iql["coins"]).items():
            coins["IQL"][aid].append(pct)
        for aid, pct in compute_coin_percentages(stats_vdn["coins"]).items():
            coins["VDN"][aid].append(pct)

        print(f"Episode {ep + 1}/{episodes} finished.")

    env1.out.release()
    env2.out.release()
    pygame.quit()

    stats = {
        algo: {
            'avg_reward': np.mean(rewards[algo]),
            'avg_length': np.mean(lengths[algo]),
            'per_agent_coin_percent': {aid: np.mean(v) for aid, v in coins[algo].items()}
        } for algo in ['IQL', 'VDN']
    }
    return stats