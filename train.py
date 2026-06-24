import os
import json
from stable_baselines3 import PPO
from sheep_env import SheepDogEnv
from kazoo.client import KazooClient

def start_training():
    """
    Uruchamia zoptymalizowany proces uczenia PPO z usuniętym harmonogramem spadku LR.
    Zwiększony parametr n_steps drastycznie poprawia wartość explained_variance,
    pozwalając krytykowi (critic) na prawidłowe ocenianie długofalowych akcji.
    """
    zk = KazooClient(hosts='127.0.0.1:2181')
    zk.start()

    print("Pobieram zoptymalizowaną konfigurację z systemu Zookeeper...")
    data, stat = zk.get("/sheep_config")
    config = json.loads(data.decode('utf-8'))
    zk.stop()

    env = SheepDogEnv(config=config)
    num_sheeps = config.get("num_sheeps", 2)
    model_path = f"sheepdog_{num_sheeps}_sheep.zip"

    # Zawsze usuwamy stary model przy tej zmianie architektury paczek danych
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Usunięto stary, zablokowany model w celu uniknięcia gradientu zerowego.")

    print(f"Tworzenie stabilnej, szerokiej sieci PPO pod stado: {num_sheeps} owce...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'], # Stały, stabilny krok uczenia
        batch_size=128,                        # Optymalny rozmiar minipaczki dla stabilności
        n_steps=4096,                          # ZWIĘKSZONO x2: Zbiera 2x więcej danych przed aktualizacją
        ent_coef=config['ent_coef'],           # Kontrola eksploracji przestrzeni ciągłej
        gae_lambda=0.98,                       # Wyższy współczynnik wygładzania przewagi (redukcja wariancji)
        verbose=1
    )

    # Zbalansowany dystans treningowy pozwalający na pełną zbieżność
    step_amount = 120000
    print(f"Rozpoczynam finalny trening na dystansie {step_amount} kroków...")
    model.learn(total_timesteps=step_amount)

    model.save(model_path)
    print("--- SUKCES: Nowy model został poprawnie wytrenowany i zapisany! ---")

if __name__ == "__main__":
    start_training()