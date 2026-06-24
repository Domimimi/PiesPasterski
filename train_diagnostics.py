import json
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sheep_env import SheepDogEnv


class AcademicDiagnosticsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(AcademicDiagnosticsCallback, self).__init__(verbose)
        self.total_episodes = 0
        self.global_sheep_saved = 0
        self.global_perfect_runs = 0
        self.dog_velocities = []

        self.total_steps = 0
        self.frozen_steps = 0
        self.saturated_steps = 0

    def _on_step(self) -> bool:
        if "actions" in self.locals:
            raw_actions = self.locals["actions"]
            for act in raw_actions:
                self.total_steps += 1
                velocity = np.linalg.norm(act)
                self.dog_velocities.append(velocity)

                if velocity < 0.15:
                    self.frozen_steps += 1
                if np.any(np.abs(act) > 0.95):
                    self.saturated_steps += 1

        for info in self.locals.get("infos", []):
            if "sheep_saved_summary" in info:
                self.total_episodes += 1
                self.global_sheep_saved += info["sheep_saved_summary"]
                self.global_perfect_runs += info["perfect_run"]

        return True

    def log_final_report(self, env):
        logger = self.model.logger
        explained_var = logger.name_to_value.get("train/explained_variance", np.nan)
        value_loss = logger.name_to_value.get("train/value_loss", np.nan)
        # POPRAWKA KLUCZA: Zmiana z policy_loss na policy_gradient_loss
        policy_loss = logger.name_to_value.get("train/policy_gradient_loss", np.nan)

        print("\n" + "=" * 60)
        print("   GLOBALNY ZAAWANSOWANY RAPORT DIAGNOSTYCZNY (RL-METRICS)")
        print("=" * 60)
        print(f"Przeanalizowano kroków środowiskowych : {self.num_timesteps}")
        print(f"Łączna liczba rozegranych epizodów : {self.total_episodes}")
        print("-" * 60)
        print(" METRYKI ZBIEŻNOŚCI SIECI (ALGORITHM convergence):")
        print(f"  Wyjaśniona Wariancja (Explained Variance) : {explained_var:.4f}")
        print(f"  Błąd Funkcji Wartości (Value Loss)       : {value_loss:.4f}")
        print(f"  Strata Polityki (Policy Loss)             : {policy_loss}")
        print("-" * 60)
        print(" METRYKI BEHAWIORALNE AGENTA:")
        avg_vel = np.mean(self.dog_velocities) if self.dog_velocities else 0.0
        freeze_ratio = (self.frozen_steps / self.total_steps * 100) if self.total_steps > 0 else 0.0
        saturation_ratio = (self.saturated_steps / self.total_steps * 100) if self.total_steps > 0 else 0.0

        print(f"  Średnia płynność działań (Action Norm)    : {avg_vel:.4f}")
        print(f"  Wskaźnik paraliżu agenta (Freeze Ratio)   : {freeze_ratio:.2f}%")
        print(f"  Przesterowanie sieci (Action Saturation)  : {saturation_ratio:.2f}%")
        print("-" * 60)
        print(" METRYKI SUKCESU ZADANIA:")
        print(f"  Suma owiec wprowadzonych podczas treningu : {self.global_sheep_saved}")
        print(f"  Liczba idealnych epizodów (100% stada)     : {self.global_perfect_runs}")

        if self.total_episodes > 0:
            success_rate = (self.global_perfect_runs / self.total_episodes) * 100
            avg_sheep = self.global_sheep_saved / self.total_episodes
            print(f"  Średnia liczba owiec na epizod            : {avg_sheep:.2f} / {env.num_sheeps}")
            print(f"  Globalny wskaźnik pełnego sukcesu (SR %)  : {success_rate:.2f}%")
        else:
            print("  Brak zakończonych epizodów w logu bufora.")
        print("=" * 60 + "\n")


def train():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("Brak pliku config.json! Uruchom najpierw config_manager.py")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    env_config = config.get("env_config", {})
    ppo_config = config.get("ppo_config", {})

    # 1. Inicjalizujemy środowisko (które ma już w sobie dynamiczne max_steps z poprzedniego kroku)
    env = SheepDogEnv(config=env_config)

    # 2. DYNAMICZNE WYLICZANIE MATEMATYCZNE (Twój wzór):
    # Pobieramy realne wartości prosto z obiektu środowiska
    num_sheeps = env.num_sheeps
    max_steps = env.max_steps

    total_timesteps = 125 * num_sheeps * max_steps

    print("-" * 60)
    print(f"[MATEMATYCZNY BUDŻET TRENINGOWY]")
    print(f"  Liczba owiec: {num_sheeps}")
    print(f"  Długość iteracji (max_steps): {max_steps}")
    print(f"  Wyliczone total_timesteps: {total_timesteps}")
    print("-" * 60)

    model_filename = "ppo_sheep_dog_stable"
    model_zip_path = f"{model_filename}.zip"

    # LOGIKA AUTO-WCZYTYWANIA MODELU
    if os.path.exists(model_zip_path):
        print(f"\n[!] Znaleziono zapisany model '{model_zip_path}'. Wczytuję do kontynuacji treningu...")
        model = PPO.load(model_filename, env=env)
        model.learning_rate = ppo_config.get("learning_rate", 5e-5)
    else:
        print("\n[!] Brak zapisanego modelu. Tworzę nową sieć od zera...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=ppo_config.get("learning_rate", 5e-5),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=10,
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            seed=42
        )

    diagnostics_callback = AcademicDiagnosticsCallback()

    # Przekazujemy dynamicznie obliczoną wartość do procesu uczenia
    print(f"Uruchamianie treningu (sesja: {total_timesteps} kroków)...")
    model.learn(total_timesteps=total_timesteps, callback=diagnostics_callback, reset_num_timesteps=False)

    diagnostics_callback.log_final_report(env)

    model.save(model_filename)
    print(f"Model został zaktualizowany i zapisany jako '{model_zip_path}'.")

if __name__ == "__main__":
    train()