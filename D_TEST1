import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import rospy
from collections import deque

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epsilon_decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.timestep = 0
        self.batch_size = 32
        self.policy_net = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = 100



    def new_environment(self):
        self.uwb_distance = rospy.Publisher('/vesc/ackermann_cmd_mux/input/uwb', int32)
        self.camera_distance = rospy.Publisher('/vesc/ackermann_cmd_mux/input/c_d', int32)
        self.rssi_distance = rospy.Publisher('/vesc/ackermann_cmd_mux/input/r_d', int32)
        #uwb_distance = np.random.randint(0, 100)  # 랜덤한 UWB 거리값 생성
        #camera_distance = np.random.randint(0, 100)  # 랜덤한 카메라 거리값 생성
        #rssi_distance = np.random.randint(0, 100)  # 랜덤한 RSSI 거리값 생성
        return (self.uwb_distance, self.camera_distance, self.rssi_distance)


    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reward_function(self, current_state, next_state,action , next_action):
        action = action*0.01
        next_action= next_action*0.01
        uwb_distance, camera_distance, rssi = current_state
        next_uwb_distance, next_camera_distance, next_rssi = next_state

      #  distance_diff = abs(next_uwb_distance - next_camera_distance)
    # reward = - abs(uwb_distance +action - camera_distance) +  (100-abs(next_uwb_distance+next_action - next_camera_distance))
        distance_diff = abs(camera_distance - uwb_distance*action)
        next_distance_diff = abs(next_camera_distance - next_uwb_distance*next_action)
        reward_next = 1 / (next_distance_diff + 1e-6)
        reward_per = 1 / (distance_diff + 1e-6)

        reward = reward_next + reward_per
        return reward

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_actions = torch.argmax(self.policy_net(next_state_batch), dim=1)
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = self.loss_fn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.timestep % 100 == 0:
            self.timestep += 1

        # Update target network
        if self.timestep % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    rospy.init_node('DDQN_D', anonymous=True)
    episodes = 100
    agent = Agent(state_dim=3, action_dim=200, hidden_dim=64, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995)
    for episode in range(episodes):
        state = agent.new_environment()
        action = agent.choose_action(state)
        done = False
        total_reward = 0
        while not done:
            next_action = agent.choose_action(state)
            next_state = agent.new_environment()

            reward = agent.reward_function(state, next_state,action ,next_action)
            agent.store_transition(state, next_action, reward, next_state, done)
            state = next_state
            action =next_action
            total_reward += reward
            agent.train()

        print("Episode: ", episode, "Total Reward: ", total_reward)
