import json
import time
from kazoo.client import KazooClient
from sheep_env import SheepDogEnv
from stable_baselines3 import PPO


def run_test():
    """Uruchomienie skryptu testującego zachowanie wyuczonego agenta w środowisku renderowanym."""
    # Pobranie konfiguracji w celu dopasowania wymiarów sieci wejściowej
    try:
        zk = KazooClient(hosts="127.0.0.1:2181")
        zk.start()
        data, _ = zk.get("/sheep_config")
        config = json.loads(data.decode("utf-8"))
        zk.stop()
    except:
        print(
            "Nie można połączyć się z Zookeeperem. Używam domyślnych parametrów testowych dla 2 owiec."
        )
        config = {
            "sheep_speed": 3.0,
            "repulsion_radius": 100.0,
            "dog_speed": 6.0,
            "num_sheeps": 2,
        }

    env = SheepDogEnv(config=config)
    num_sheeps = config.get("num_sheeps", 2)
    model_path = f"ppo_sheep_dog_stable.zip"

    try:
        model = PPO.load(model_path)
        print(f"Model dla {num_sheeps} owiec załadowany pomyślnie!")
    except:
        print(
            f"Błąd fatalny: Nie odnaleziono pliku {model_path}. Uruchom najpierw train.py!"
        )
        return

    obs, info = env.reset()
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            print("Koniec sekwencji testowej - Resetowanie środowiska.")
            time.sleep(1)
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_test()