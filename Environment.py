import numpy as np
import random
import pygame
import cv2


class CoinCollectionEnv:
    def __init__(self, grid_size=5, num_agents=1, num_coins=3, max_steps=50, cell_size=80):
        # Environment settings
        self.grid_size = grid_size
        self.num_agents = min(num_agents, 4)
        self.num_coins = min(num_coins, 5)
        self.max_steps = max_steps
        self.cell_size = cell_size

        # Coin Slots: corners and center
        self.coin_slots = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1), ((grid_size - 1) // 2, (grid_size - 1) // 2)]

        # Actions: up, down, left, right, stay
        self.action_space = [0, 1, 2, 3, 4]
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }

        # Colors
        self.bg_color = (255, 255, 255)
        self.grid_color = (200, 200, 200)
        self.coin_color = (255, 215, 0)
        self.agent_colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0), (255, 0, 255)]  # Support up to 4 agents

        # Pygame setup
        pygame.init()
        width = grid_size * cell_size
        height = grid_size * cell_size
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption("Coin Collection Environment")
        self.clock = pygame.time.Clock()

        fps = 5
        self.out = cv2.VideoWriter('Records/game.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        self.agent_positions = []
        self.coins = []
        self.steps = 0
        self.reset(num_coins)

    def reset(self, n_coins):
        self.num_coins = np.clip(n_coins, 1, 5)
        self.agent_positions = [self._random_empty_agent_pos() for _ in range(self.num_agents)]
        self.coins = []
        for _ in range(self.num_coins):
            self.coins.append(self._empty_coin_pos())
        self.steps = 0
        return self.get_obs()

    def step(self, actions):
        rewards = [0] * self.num_agents
        self.steps += 1

        # Move agents
        for i, action in enumerate(actions):
            dx, dy = self.action_map[action]
            x, y = self.agent_positions[i]
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)
            self.agent_positions[i] = (new_x, new_y)

        # Check for coin collection
        for i, pos in enumerate(self.agent_positions):
            if pos in self.coins:
                rewards[i] += 10
                self.coins.remove(pos)

        # Apply time penalty
        rewards = [r - 1 for r in rewards]

        # Check if done
        done = len(self.coins) == 0 or self.steps >= self.max_steps
        return self.get_obs(), rewards, done

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.bg_color)

        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        # Draw coins
        for (cx, cy) in self.coins:
            center = (cy * self.cell_size + self.cell_size // 2, cx * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, self.coin_color, center, self.cell_size // 4)

        # Draw agents
        for idx, (ax, ay) in enumerate(self.agent_positions):
            color = self.agent_colors[idx % len(self.agent_colors)]
            rect = pygame.Rect(ay * self.cell_size + 5, ax * self.cell_size + 5,
                               self.cell_size - 10, self.cell_size - 10)
            pygame.draw.rect(self.screen, color, rect)

        # Capture Frame
        frame = pygame.surfarray.array3d(self.screen)
        frame = frame.swapaxes(0, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.out.write(frame)

        pygame.display.flip()
        self.clock.tick(5)  # 5 FPS

    def _random_empty_agent_pos(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in getattr(self, 'agent_positions', []) and pos not in getattr(self, 'coin_slots', []):
                return pos

    def _empty_coin_pos(self):
        free = [pos for pos in self.coin_slots if pos not in self.coins]
        if not free:
            print('None')
            return None
        pos = random.choice(free)
        return pos

    def _get_coins_mask(self):
        mask = []
        for i, slot in enumerate(self.coin_slots):
            if slot in self.coins:
                mask.append(1)
            else:
                mask.append(0)
        return mask

    def get_obs(self):
        xs = [pos[0] for pos in self.agent_positions]
        ys = [pos[1] for pos in self.agent_positions]
        return xs, ys, self._get_coins_mask()
