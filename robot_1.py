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
import numpy as np
import ballbeam_gym
import matplotlib.pyplot as plt
import os.path
import cv2

#from Arm_Lib import Arm_Device
#Arm = Arm_Device()

#Image Rec
cam= cv2.VideoCapture(0)
previous_x=0
timenow=0
#Image rec

#Bring the robot to the initial position
def robot_init():

    Arm.Arm_serial_servo_write(1, 90, 200)
    time.sleep(0.1)
    Arm.Arm_serial_servo_write(2, 90, 500)
    time.sleep(0.1)
    Arm.Arm_serial_servo_write(3, 90, 500)
    time.sleep(0.1)
    Arm.Arm_serial_servo_write(4,  0, 200)
    time.sleep(0.1)
    Arm.Arm_serial_servo_write(5, 90, 200)
    time.sleep(0.1)

def robot():

    angle = 0
    vel = 0
    pos_x =0

def image_rec_start():
    #HSV Range for RED 
    lowerbound=np.array([170,70,50])
    upperbound=np.array([180,255,255])
    # Getting image from video
    ret, img=cam.read()
    # Resizing the image
    img=cv2.resize(img,(400,300))
    # Smoothning image using GaussianBlur
    imgblurred=cv2.GaussianBlur(img,(11,11),0)
    # Converting image to HSV format
    imgHSV=cv2.cvtColor(imgblurred,cv2.COLOR_BGR2HSV) #source:https://thecodacus.com/opencv-object-tracking-colour-detection-python/#.Wz9tQN6Wl_k
    # Masking red color
    mask=cv2.inRange(imgHSV,lowerbound,upperbound) #source:https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
    # Removing Noise from the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Extracting contour
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Drawing Contour		
    cv2.drawContours(img,cnts,-1,(255,0,0),3)
    #Processing each contour
    for c in cnts:  #source: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    # compute the center of the  maximum area contour
        m=max(cnts,key=cv2.contourArea) #finding the contour with maximum area
        M = cv2.moments(m)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Drawing the max area contour and center of the shape on the image
        cv2.drawContours(img, [m], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #Drawing a vertical central line with RED color(BGR)
        cv2.line(img,(200,0),(200,300),(0,0,255),2)
        #Drawing a vertical line at the centre with Blue color
        cv2.line(img,(cX,0),(cX,300),(255,0,0),2)
        #Displaying mask
        cv2.imshow("mask",mask)
        #Displaying image
        cv2.imshow("cam",img)
        desired_pos = 200
        #PID calcuation
        pos_x = (cX-desired_pos)/1.5 #parametro a metrificar
        global timenow, previous_x
        time_previous=timenow
        timenow=time.time()
        elapsedTime=timenow-time_previous
        vel_x = ((pos_x-previous_x)/elapsedTime)
        previous_x=pos_x
        #arm_angle = Arm.Arm_serial_servo_read(5);
        arm_angle = 0
        out = np.zeros(4)
        return np.array([arm_angle, pos_x, vel_x, 0])
        

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
kwargs = {'timestep': 0.05, 
          'setpoint': 0.00,
          'beam_length': 1.0,
          'max_angle': 0.4, #isso é um limitante
          'max_timesteps':None,
          'init_velocity': 0.7,
          'action_mode': 'discrete'} #discrete chose (keep, increase, decrease)

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
    
# Main Beggining here 
#robot_init()



qnet_agent = QNet_Agent()

steps_total = []

frames_total = 0 
solved_after = 0
solved = False


start_time = time.time()

for i_episode in range(num_episodes):
    env.reset()
    env.env.state = image_rec_start()
    state = env.reset()
   # print(state)
    #print(state)
   # state = image_rec_start()
    step = 0
    while True:
       # print("Camera Feedback vector"+str(image_rec_start()))
        print(state)
        step += 1
        frames_total += 1
        
        if i_episode % 1 == 0: #Print episodes
            env.render()
           
       # print("obs: "+str(state))
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state)
        
        new_state, reward, done, info = env.step(action)

        #print da ação
    
        #Arm.Arm_serial_servo_write(5, env.step(action)[0][0]*90,1)
        state = image_rec_start()
        #print(env.step(action)[0][0]*90)
        #state = new_state
        
        
        if done:
            steps_total.append(step)
            
            mean_reward_100 = sum(steps_total[-100:])/100
            
            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0):
                
                
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \n frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(steps_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(steps_total)/len(steps_total),
                   
                    frames_total
                          ) 
                  )
                  
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



            break
        

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
