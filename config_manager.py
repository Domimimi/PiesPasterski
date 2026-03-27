from kazoo.client import KazooClient
import json


def setup_config():
    zk = KazooClient(hosts='127.0.0.1:2181')
    zk.start()

    #parametry symulacji
    config_data = {
        "sheep_speed": 3.0,
        "dog_speed": 6.0,
        "repulsion_radius": 100.0,
        "learning_rate": 0.0001,
        "ent_coef": 0.005
    }

    #zapisanie w zookeeperze
    if not zk.exists("/sheep_config"):
        zk.create("/sheep_config", json.dumps(config_data).encode('utf-8'))
    else:
        zk.set("/sheep_config", json.dumps(config_data).encode('utf-8'))

    zk.stop()


if __name__ == "__main__":
    setup_config()
    print("Konfiguracja wysłana do Zookeepera!")