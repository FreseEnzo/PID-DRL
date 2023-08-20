#
# Dueling DQN g
# PIBIC 2D DQN
# Coded by Enzo Frese
#

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time
import ballbeam_gym
import matplotlib.pyplot as plt
import os.path

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

plt.style.use('ggplot')

# Saving Model Variabes
file2save = 'treino_robot.pth'
save_model_frequency = 10000
resume_previous_training = True



#Environment initial definitions
kwargs = {'timestep': 0.1, # Tempo entre as ações 
          'setpoint': 0.0, # Ponto de equilíbrio desejado
          'beam_length': 1.0, # Tamanho da bola
          'max_angle': 0.4, # Ângulo máximo em radianos
          'max_timesteps':None, # None, pois não é um treinamento
          'init_velocity': 1.0, # Velocidade inicial
          'action_mode': 'discrete'} # Modo de ação discreto

#Building the environment 
env = gym.make('BallBeamSetpoint-v0', **kwargs)

#Saving Video
#directory = './DRLVideos/'
#env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%20==0)

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)


##TYPE
training = False
###### PARAMS ######
learning_rate = 0.001
num_episodes = 1000
gamma = 1

hidden_layer = 64

replay_mem_size = 50000
batch_size = 32

update_target_frequency = 10000

double_dqn = True

egreedy = 0.1
egreedy_final = 0.01
egreedy_decay = 500

report_interval = 1
score_to_solve = 100

clip_error = False

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n


def load_model():
    return torch.load(file2save)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        
        self.advantage = nn.Linear(hidden_layer,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)
        
        self.value = nn.Linear(hidden_layer,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        
        
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        
        output_advantage = self.advantage(output1)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        
        output_value = self.value(output1)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)
        
        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final
    
class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.update_target_counter = 0

        if resume_previous_training and os.path.exists(file2save):
          print("Loading previously saved model ... ")
          self.nn.load_state_dict(load_model())

        self.number_of_frames = 0

    def select_action(self,state):
        with torch.no_grad():
            
            state = Tensor(state).to(device)
            action_from_nn = self.nn(state)
            action = torch.max(action_from_nn,0)[1]
            action = action.item()        
       
        return action
    
    
qnet_agent = QNet_Agent()

steps_total = []

frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    #for step in range(100):
    while True:
        
        step += 1
        frames_total += 1

        if i_episode % 1 == 0: #Print episodes
          env.render()
        
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state)
        
        new_state, reward, done, info = env.step(action)
        
        state = new_state
        

print("\n\n\n\nAverage reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))
if solved:
    print("Solved after %i episodes" % solved_after)

plt.figure(figsize=(12,5))
plt.title("Rewards")
#plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.plot(steps_total, alpha=0.6, color='red')
plt.show()
plt.savefig("Rewards.png")
plt.close()

env.close()
env.env.close()
