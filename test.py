import time
from stable_baselines3 import PPO
from sheep_env import SheepDogEnv

#tworzenie takiego samego środowiska
config = {"sheep_speed": 3.0, "repulsion_radius": 100.0, "dog_speed": 6.0}
env = SheepDogEnv(config=config)

#wczytanie wytrenowanego modelu (plik .zip)
try:
    model = PPO.load("sheepdog_final_model")
    print("Model wczytany")
except:
    print("Błąd: Nie znaleziono pliku sheepdog_final_model.zip")

def run_test():
    # podgląd
    obs, info = env.reset()
    for i in range(5000):
        # model decyduje o ruchu na podstawie obserwacji
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # wyświetlenie okna
        env.render()

        if terminated or truncated:
            print("Koniec rundy")
            time.sleep(1)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    run_test()