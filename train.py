import os
from stable_baselines3 import PPO
from sheep_env import SheepDogEnv
from kazoo.client import KazooClient
import json

def start_training():
    # pobranie parametrów z Zookeepera
    zk = KazooClient(hosts='127.0.0.1:2181')
    zk.start()

    print("Pobieram konfigurację z Zookeepera")
    data, stat = zk.get("/sheep_config")
    config = json.loads(data.decode('utf-8'))

    # zamyknięcie połączenia
    zk.stop()

    # tworzenie środowiska
    env = SheepDogEnv(config=config)

    model_path = "sheepdog_final_model.zip"

    # tworzenie modelu AI (PPO)
    if os.path.exists(model_path):
        print("Znaleziono istniejący model")
        # wczytanie modelu i podpięcie go pod aktualne środowisko
        model = PPO.load(model_path, env=env, learning_rate=config['learning_rate'], ent_coef=config['ent_coef'])
    else:
        print("Brak zapisanego modelu")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config['learning_rate'],
            ent_coef=config['ent_coef'],
            batch_size=256,
            n_steps=4096,
            verbose=1
        )

    # rozpoczęcie nauki
    step_amount = 200000
    print(f"Rozpoczynam trening psa na {step_amount} kroków")
    model.learn(total_timesteps=step_amount)

    # zapisanie wytrenowanego psa
    model.save(model_path)

if __name__ == "__main__":
    start_training()