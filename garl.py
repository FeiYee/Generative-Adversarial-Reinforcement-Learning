import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 设定随机种子以保持结果的一致性
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Deep Q-Learning Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# GAN的生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        return torch.tanh(self.fc2(x))

# GAN的鉴别器
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class GARL:
    def __init__(self, state_size, action_size, generator_input_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)

        self.dqn = DQN(state_size, action_size)
        self.generator = Generator(generator_input_size, state_size)
        self.discriminator = Discriminator(state_size)

        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.dqn(torch.tensor(next_state, dtype=torch.float))).item()
            target_f = self.dqn(torch.tensor(state, dtype=torch.float))
            target_f[0][action] = target
            self.dqn_optimizer.zero_grad()
            loss = nn.MSELoss()(self.dqn(torch.tensor(state, dtype=torch.float)), target_f)
            loss.backward()
            self.dqn_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_generator_discriminator(self):
        # 训练鉴别器
        real_data = torch.tensor(np.random.randn(self.batch_size, self.state_size), dtype=torch.float)
        fake_data = self.generator(torch.tensor(np.random.randn(self.batch_size, self.state_size), dtype=torch.float))
        real_labels = torch.ones(self.batch_size, 1)
        fake_labels = torch.zeros(self.batch_size, 1)

        self.discriminator_optimizer.zero_grad()
        outputs_real = self.discriminator(real_data)
        outputs_fake = self.discriminator(fake_data.detach())
        loss_real = nn.BCELoss()(outputs_real, real_labels)
        loss_fake = nn.BCELoss()(outputs_fake, fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        self.discriminator_optimizer.step()

        # 训练生成器
        self.generator_optimizer.zero_grad()
        outputs = self.discriminator(fake_data)
        g_loss = nn.BCELoss()(outputs, real_labels)
        g_loss.backward()
        self.generator_optimizer.step()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.dqn(torch.tensor(state, dtype=torch.float))
        return torch.argmax(act_values[0]).item()
