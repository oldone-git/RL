import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

class TestArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.debug_line = None

    def reset(self, seed=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        for i in range(p.getNumJoints(self.robot)):
            if i not in [0, 3]:
                p.setJointMotorControl2(self.robot, i, p.VELOCITY_CONTROL, force=0)
                p.resetJointState(self.robot, i, 0)

        self.target_pos = [0.5, 0.0, 0.3]
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        self.target = p.createMultiBody(0, vis_id, basePosition=self.target_pos)

        p.resetJointState(self.robot, 0, 0)
        p.resetJointState(self.robot, 3, 0)
        self.last_distance = None

        return self._get_obs(), {}

    def _get_obs(self):
        joints = [p.getJointState(self.robot, j)[0] for j in [0, 3]]
        end_pos = p.getLinkState(self.robot, 6)[0]

        if self.debug_line:
            p.removeUserDebugItem(self.debug_line)
        self.debug_line = p.addUserDebugLine(
            end_pos, self.target_pos, [0, 1, 0], lineWidth=2, lifeTime=0.1)

        return np.array([
            *joints,
            end_pos[0], end_pos[1],
            self.target_pos[0], self.target_pos[1]
        ])

    def step(self, action):
        scaled = action * (np.pi / 4)
        for i, joint in enumerate([0, 3]):
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                    targetPosition=scaled[i], force=1000)

        for _ in range(5): p.stepSimulation()

        obs = self._get_obs()
        end_pos = p.getLinkState(self.robot, 6)[0]
        distance = np.linalg.norm(np.array(end_pos[:2]) - np.array(self.target_pos[:2]))

        reward = -distance
        if self.last_distance is not None:
            reward += (self.last_distance - distance) * 10
        self.last_distance = distance

        done = distance < 0.1
        if done:
            print(f" Попадание! Дистанция: {distance:.3f}")
            reward += 50
        time.sleep(1 / 240)
        return obs, reward, done, False, {}

# --- Загрузка модели ---
model_path = "models/simplified_arm"
if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(" Модель не найдена!")

env = TestArmEnv()
model = PPO.load(model_path, env=env)
obs, _ = env.reset()

print("👁 Нажмите Ctrl+C для остановки")

try:
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
except KeyboardInterrupt:
    print(" Завершено пользователем")
finally:
    p.disconnect()
