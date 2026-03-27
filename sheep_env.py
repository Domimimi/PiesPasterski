import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SheepDogEnv(gym.Env):
    #konstruktor środowiska
    def __init__(self, config=None):
        super(SheepDogEnv, self).__init__()

        #konfiguracja świata
        if config is None:
            config = {"sheep_speed": 3.0, "repulsion_radius": 100.0, "dog_speed": 5.0}

        self.map_size = 600
        self.sheep_speed = config["sheep_speed"]
        self.dog_speed = config["dog_speed"]
        self.repulsion_radius = config["repulsion_radius"]

        self.max_steps = 800
        self.current_step = 0

        #akcja psa
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        #obserwacja psa
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        #stan początkowy
        self.goal_pos = np.array([550, 550], dtype=np.float32)
        self.dog_pos = None
        self.sheep_pos = None

        self.last_dist_sheep_goal = 0.0

        #renderowanie
        self.render_mode = 'human'
        self.screen = None
        self.clock = None

    def _get_obs(self):
        #normalizacja pozycji do zakresu [0, 1]
        return np.concatenate([
            self.dog_pos / self.map_size,
            self.sheep_pos / self.map_size,
            self.goal_pos / self.map_size
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        #losowanie pozycji (z dala od bramek na start)
        self.dog_pos = np.random.uniform(50, 400, size=(2,)).astype(np.float32)
        self.sheep_pos = np.random.uniform(100, 450, size=(2,)).astype(np.float32)

        self.last_dist_sheep_goal = np.linalg.norm(self.sheep_pos - self.goal_pos)

        return self._get_obs(), {}

    def step(self, action):
        #ruch Psa
        self.dog_pos += action * self.dog_speed
        self.dog_pos = np.clip(self.dog_pos, 0, self.map_size)

        #ruch Owcy (ucieczka przed psem)
        dist_dog_sheep = np.linalg.norm(self.dog_pos - self.sheep_pos)
        if dist_dog_sheep < self.repulsion_radius:
            run_direction = self.sheep_pos - self.dog_pos
            run_direction /= (dist_dog_sheep + 1e-6)
            self.sheep_pos += run_direction * self.sheep_speed

        self.sheep_pos = np.clip(self.sheep_pos, 0, self.map_size)

        #obliczenia dystansów
        dist_sheep_goal = np.linalg.norm(self.sheep_pos - self.goal_pos)

        #NAGRODY
        reward = 0.0

        #nagroda za samo bycie blisko owcy (motywacja do interakcji)
        #im bliżej psa do owcy (do pewnego momentu), tym lepiej
        if dist_dog_sheep < 150:
            reward += 0.1
        if dist_dog_sheep < self.repulsion_radius:
            reward += 0.2

        #nagroda za pchanie do celu
        dist_gain = self.last_dist_sheep_goal - dist_sheep_goal
        if dist_gain > 0:
            reward += dist_gain * 2.0

        self.last_dist_sheep_goal = dist_sheep_goal

        #kara za czas
        reward -= 0.01

        terminated = False
        if dist_sheep_goal < 45:
            reward += 500.0
            terminated = True
            print("SUKCES")

        #kara za drżenie
        action_penalty = 0.001 * np.sum(np.square(action))
        reward -= action_penalty

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_size, self.map_size))
            pygame.display.set_caption("Sheep Dog AI")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((30, 150, 30))  # Trawa

        #bramka
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal_pos.astype(int), 45, 2)

        #pies
        pygame.draw.rect(self.screen, (0, 0, 255),
                         (int(self.dog_pos[0] - 10), int(self.dog_pos[1] - 10), 20, 20))

        #owca
        pygame.draw.circle(self.screen, (255, 255, 255), self.sheep_pos.astype(int), 10)

        pygame.display.flip()
        self.clock.tick(60)