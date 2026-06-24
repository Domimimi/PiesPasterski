import json
import os
import time
from sheep_env import SheepDogEnv
from stable_baselines3 import PPO


def run_test():
    """Uruchomienie skryptu testującego zachowanie wyuczonego agenta w środowisku renderowanym."""
    config_path = "config.json"

    # Próba wczytania konfiguracji z pliku config.json (tak jak w treningu)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
        # Pobieramy sekcję env_config
        config = config_data.get("env_config", {})
        print(f"[+] Załadowano konfigurację z pliku '{config_path}'. Liczba owiec: {config.get('num_sheeps', 3)}")
    else:
        # Całkowity fallback, gdyby pliku json nie było - ustawiamy bezpieczne 3 owce i spójne prędkości
        print(f"[-] Brak pliku {config_path}! Używam awaryjnych parametrów dla 3 owiec.")
        config = {
            "sheep_speed": 3.0,
            "repulsion_radius": 115.0,
            "dog_speed": 4.5,
            "num_sheeps": 3,
        }

    env = SheepDogEnv(config=config)
    num_sheeps = config.get("num_sheeps", 3)
    model_filename = "ppo_sheep_dog_stable"
    model_zip_path = f"{model_filename}.zip"

    if not os.path.exists(model_zip_path):
        print(f"[!] Błąd fatalny: Nie odnaleziono pliku modelu '{model_zip_path}'. Uruchom najpierw trening!")
        return

    try:
        # Wczytujemy model przypisując go jawnie do nowo utworzonego środowiska testowego
        model = PPO.load(model_filename, env=env)
        print(f"[+] Model dla {num_sheeps} owiec załadowany pomyślnie! Rozpoczynam wizualizację...")
    except Exception as e:
        print(f"[!] Błąd ładowania modelu (prawdopodobnie niedopasowanie wymiarów sieci): {e}")
        return

    obs, info = env.reset()
    for _ in range(5000):
        # deterministic=True, ponieważ podczas testów chcemy czystej strategii, a nie eksploracji szumem
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            print("Koniec sekwencji testowej (Sukces lub Limit kroków) - Resetowanie środowiska.")
            time.sleep(1)
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_test()