import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sheep_env import SheepDogEnv


# Definiujemy cel optymalizacji (tzw. "Objective Function")
def objective(trial):
    # 1. Definiujemy przestrzenie wyszukiwania dla parametrów
    # Optuna sama wybierze wartości z podanych zakresów dla każdej próby
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])

    # 2. Tworzymy środowisko (na sztywno z bazową konfiguracją prędkości)
    config = {"sheep_speed": 3.0, "repulsion_radius": 100.0, "dog_speed": 6.0}
    env = SheepDogEnv(config=config)

    # 3. Inicjalizujemy model PPO z parametrami wylosowanymi przez Optunę
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=0  # Wyłączamy standardowe logi SB3, żeby nie zaśmiecać konsoli
    )

    # 4. Uruchamiamy KRÓTKI trening (np. 40 000 kroków)
    # Chcemy tylko zobaczyć, który model wykazuje najszybszy start i najlepszy potencjał
    print(
        f"\n[Próba {trial.number}] Testuję: LR={learning_rate:.5f}, Ent={ent_coef:.5f}, Batch={batch_size}, Steps={n_steps}")
    model.learn(total_timesteps=40000)

    # 5. Oceniamy model (wyznaczamy średnią nagrodę z 5 testowych rund)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    # Optuna dąży do MAKSYMALIZACJI tej zwracanej wartości
    return mean_reward


if __name__ == "__main__":
    print("=== START AUTOMATYCZNEGO DOBIERANIA PARAMETRÓW (OPTUNA) ===")

    # Tworzymy badanie (study) nakierowane na maksymalizację nagrody
    study = optuna.create_study(direction="maximize")

    # Uruchamiamy optymalizację na 15 prób (trials)
    # Każda próba to 40k kroków, więc 15 prób potrwa około 15-20 minut.
    study.optimize(objective, n_trials=15)

    print("\n" + "=" * 40)
    print("🎯 OPTYMALIZACJA ZAKOŃCZONA SUKCESEM!")
    print(f"Najlepsza osiągnięta średnia nagroda: {study.best_value:.2f}")
    print("Najlepsze parametry dla Twojego psa:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
    print("=" * 40)