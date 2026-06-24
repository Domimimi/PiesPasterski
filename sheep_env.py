import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SheepDogEnv(gym.Env):
    """
    Środowisko Pasterskie - Wersja V6 'Stateczny Pasterz'.
    Stabilizuje funkcję nagrody poprzez pełną normalizację składowych progresu.
    """

    def __init__(self, config=None):
        super(SheepDogEnv, self).__init__()

        if config is None:
            config = {"sheep_speed": 3.0, "repulsion_radius": 115.0, "dog_speed": 4.5, "num_sheeps": 2}

        self.map_size = 600
        self.sheep_speed = config.get("sheep_speed", 3.0)
        self.dog_speed = config.get("dog_speed", 4.5)
        self.repulsion_radius = config.get("repulsion_radius", 115.0)
        self.num_sheeps = config.get("num_sheeps", 2)

        self.max_steps = 1000
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_shape = 2 + (self.num_sheeps * 4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_shape,), dtype=np.float32)

        self.goal_pos = np.array([550, 550], dtype=np.float32)
        self.goal_radius = 80.0

        self.dog_pos = None
        self.sheep_pos = None
        self.global_best_dists = []
        self.scored_sheep = set()
        self.episode_sheep_saved = 0

        self.render_mode = "human"
        self.screen = None
        self.clock = None

    def _get_obs(self):
        to_goal = (self.goal_pos - self.dog_pos) / self.map_size

        sheep_data = []
        for idx in range(self.num_sheeps):
            s_pos = self.sheep_pos[idx]
            is_scored = idx in self.scored_sheep
            dist_to_dog = np.linalg.norm(s_pos - self.dog_pos)
            sheep_data.append({
                'pos': s_pos.copy(),
                'is_scored': is_scored,
                'dist_to_dog': dist_to_dog
            })

        # Sortowanie: Aktywne owce na początek (od najbliższej), bezpieczne na koniec
        sheep_data.sort(key=lambda x: (x['is_scored'], x['dist_to_dog']))

        flat_sheep_obs = []
        for data in sheep_data:
            s_pos = data['pos']
            if data['is_scored']:
                to_sheep = (self.goal_pos - self.dog_pos) / self.map_size
                to_herding_point = (self.goal_pos - self.dog_pos) / self.map_size
            else:
                to_sheep = (s_pos - self.dog_pos) / self.map_size

                from_goal_to_sheep = s_pos - self.goal_pos
                dir_away = from_goal_to_sheep / (np.linalg.norm(from_goal_to_sheep) + 1e-6)

                ideal_herding_point = s_pos + dir_away * (self.repulsion_radius * 0.8)
                ideal_herding_point = np.clip(ideal_herding_point, 2.0, self.map_size - 2.0)

                to_herding_point = (ideal_herding_point - self.dog_pos) / self.map_size

            flat_sheep_obs.extend(to_sheep)
            flat_sheep_obs.extend(to_herding_point)

        obs = np.concatenate([to_goal, flat_sheep_obs]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.scored_sheep.clear()
        self.episode_sheep_saved = 0

        self.dog_pos = np.random.uniform(50, 180, size=(2,)).astype(np.float32)
        self.sheep_pos = np.random.uniform(250, 420, size=(self.num_sheeps, 2)).astype(np.float32)
        self.global_best_dists = [np.linalg.norm(s_pos - self.goal_pos) for s_pos in self.sheep_pos]

        return self._get_obs(), {}

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        reward = 0.0
        action = np.clip(action, -1.0, 1.0)

        self.dog_pos += action * self.dog_speed
        self.dog_pos = np.clip(self.dog_pos, 0.0, self.map_size)

        for idx in range(self.num_sheeps):
            if idx in self.scored_sheep:
                self.sheep_pos[idx] = self.goal_pos.copy()
                continue

            dist_dog_sheep = np.linalg.norm(self.dog_pos - self.sheep_pos[idx])
            if dist_dog_sheep < self.repulsion_radius:
                run_direction = self.sheep_pos[idx] - self.dog_pos
                run_direction /= (dist_dog_sheep + 1e-6)
                self.sheep_pos[idx] += run_direction * self.sheep_speed

            self.sheep_pos[idx] = np.clip(self.sheep_pos[idx], 0.0, self.map_size)

        current_dists = [np.linalg.norm(s_pos - self.goal_pos) for s_pos in self.sheep_pos]

        for idx in range(self.num_sheeps):
            if idx in self.scored_sheep:
                continue

            if current_dists[idx] < self.goal_radius:
                self.scored_sheep.add(idx)
                self.sheep_pos[idx] = self.goal_pos.copy()
                self.episode_sheep_saved += 1
                reward += 10.0  # Zbalansowana nagroda główna
                print(f"-> Sukces: Owca {idx + 1} w zagrodzie!")
                continue

            # KLUCZOWA POPRAWKA: Progres znormalizowany rozmiarem mapy
            if current_dists[idx] < self.global_best_dists[idx]:
                progress = self.global_best_dists[idx] - current_dists[idx]
                reward += (progress / self.map_size) * 5.0  # Stabilna skala (max 5.0 za całą drogę)
                self.global_best_dists[idx] = current_dists[idx]

            # Kary/Nagrody pomocnicze (zredukowane do mikro-bodźców)
            vec_goal_to_sheep = self.sheep_pos[idx] - self.goal_pos
            dir_goal_to_sheep = vec_goal_to_sheep / (np.linalg.norm(vec_goal_to_sheep) + 1e-6)

            vec_sheep_to_dog = self.dog_pos - self.sheep_pos[idx]
            dist_dog_sheep = np.linalg.norm(vec_sheep_to_dog)
            dir_sheep_to_dog = vec_sheep_to_dog / (dist_dog_sheep + 1e-6)

            alignment = np.clip(np.dot(dir_goal_to_sheep, dir_sheep_to_dog), -1.0, 1.0)

            ideal_point = self.sheep_pos[idx] + dir_goal_to_sheep * (self.repulsion_radius * 0.8)
            ideal_point = np.clip(ideal_point, 2.0, self.map_size - 2.0)
            dist_to_ideal = np.linalg.norm(ideal_point - self.dog_pos)

            reward += 0.02 * (1.0 - (dist_to_ideal / self.map_size))
            if alignment > 0.707:
                reward += 0.005

        reward -= 0.005  # Delikatna presja czasu

        terminated = False
        if len(self.scored_sheep) == self.num_sheeps:
            reward += 20.0
            terminated = True
            print(f"--- SUKCES KOMPLETNY ---")

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {}
        if terminated or truncated:
            info["sheep_saved_summary"] = self.episode_sheep_saved
            info["perfect_run"] = 1 if len(self.scored_sheep) == self.num_sheeps else 0

        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_size, self.map_size))
            pygame.display.set_caption("Pies Pasterski v6 - Stable Mode")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()

        self.screen.fill((30, 150, 30))
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal_pos.astype(int), int(self.goal_radius), 2)
        pygame.draw.rect(self.screen, (0, 0, 255), (int(self.dog_pos[0] - 10), int(self.dog_pos[1] - 10), 20, 20))

        for idx, s_pos in enumerate(self.sheep_pos):
            color = (20, 50, 150) if idx in self.scored_sheep else (255, 255, 255)
            pygame.draw.circle(self.screen, color, s_pos.astype(int), 10)

            if idx not in self.scored_sheep:
                from_goal = s_pos - self.goal_pos
                dir_away = from_goal / (np.linalg.norm(from_goal) + 1e-6)
                ip = s_pos + dir_away * (self.repulsion_radius * 0.8)
                ip = np.clip(ip, 2.0, self.map_size - 2.0)
                pygame.draw.circle(self.screen, (200, 0, 0), ip.astype(int), 3)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None