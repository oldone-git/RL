from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from train_arm import SimpleArmEnv  # если класс в отдельном файле
import pybullet as p

# Инициализируем окружение снова
env = Monitor(SimpleArmEnv(), filename=None)

# Загружаем сохранённую модель
model = PPO.load("models/simplified_arm", env=env)

print(" Продолжаем обучение ещё 100 000 шагов...")
model.learn(total_timesteps=100_000)
model.save("models/simplified_arm")
print(" Дообучено и сохранено!")

p.disconnect()
