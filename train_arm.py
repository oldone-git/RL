import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

class SimpleArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        p.connect(p.DIRECT)  # Без GUI для обучения
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.success_count = 0
        self.episode_steps = 0
        self.max_episode_steps = 100

    def reset(self, seed=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        for i in range(p.getNumJoints(self.robot)):
            if i not in [0, 3]:
                p.setJointMotorControl2(self.robot, i, p.VELOCITY_CONTROL, force=0)
                p.resetJointState(self.robot, i, 0)

        # Ближе цель = проще на старте обучения
        self.target_pos = [0.3, 0.0, 0.3]
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, vis_id, basePosition=self.target_pos)

        p.resetJointState(self.robot, 0, 0)
        p.resetJointState(self.robot, 3, 0)

        self.last_distance = None
        self.episode_steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        joints = [p.getJointState(self.robot, j)[0] for j in [0, 3]]
        eff = p.getLinkState(self.robot, 6)[0]
        return np.array([
            *joints,
            eff[0], eff[1],
            self.target_pos[0], self.target_pos[1]
        ])

    def step(self, action):
        self.episode_steps += 1

        scaled = action * (np.pi / 4)
        for i, j in enumerate([0, 3]):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=scaled[i], force=1000)

        for _ in range(5):
            p.stepSimulation()

        obs = self._get_obs()
        eff_pos = p.getLinkState(self.robot, 6)[0]
        distance = np.linalg.norm(np.array(eff_pos[:2]) - np.array(self.target_pos[:2]))

        # Мягкий shaping-ревард со штрафом за расстояние
        reward = -distance
        if self.last_distance is not None:
            reward += (self.last_distance - distance) * 30
        self.last_distance = distance

        success = (distance < 0.1)
        if success:
            reward += 50
            self.success_count += 1
            done = True
        elif self.episode_steps >= self.max_episode_steps:
            done = True
        else:
            done = False

        return obs, reward, done, False, {}

# 🛠 Настройки
model_path = "models/simplified_arm"
log_dir = "logs"
os.makedirs("models", exist_ok=True)

# Подключаем Monitor для логов и ep_rew_mean
env = Monitor(SimpleArmEnv(), filename=None)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

print(" Обучение начинается...")
model.learn(total_timesteps=200_000)
model.save(model_path)
print(f"Модель сохранена: {model_path}.zip")

# 🔢 Сколько целей было достигнуто?
# Достаём из env
raw_env = env.env  # эта строчка правильная!
print(f"\n Всего успешных достижений цели за обучение: {raw_env.success_count}")


# Отключаем физику
p.disconnect()
