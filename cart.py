from stable_baselines3 import PPO
import gymnasium as gym

# Среда с отрисовкой (render_mode можно убрать, если не нужно окно)
env = gym.make("CartPole-v1", render_mode="human")

# Задаём PPO-модель с логированием в TensorBoard
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs_cartpole")

# Обучение
model.learn(total_timesteps=100000)

# Тест после обучения
obs, _ = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
