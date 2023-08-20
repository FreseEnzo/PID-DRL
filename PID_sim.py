import gym
import ballbeam_gym
import numpy as np
# Pass env arguments as kwargs
kwargs = {'timestep': 0.05, 
          'setpoint': 0.0,
          'beam_length': 1.0,
          'max_angle': 0.4,
          'init_velocity': 1.6,
          'action_mode': 'continuous'}

# Create env
env = gym.make('BallBeamSetpoint-v0', **kwargs)

# Constants for PID calculation
Kp = 2.0
Kd = 1.0
env.reset()
# Simulate 1000 steps
for i in range(1000):   
    # Control theta with a PID controller
    env.render()
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, obs = env.step(theta)

    if done:
        env.reset()
