import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

device   = torch.device("cpu")

class PG(nn.Module):
    def __init__(self, num_inputs, num_action, num_hidden = 64, device= device):
        super(PG, self).__init__()
        self.device  = device
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_action)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state       = torch.FloatTensor(state).unsqueeze(0).to(self.device) # to vector
        action_prob = self.forward(state)
        return action_prob.detach().cpu().numpy()[0]

    def learn(self, S, A, R, GAMMA):
        # calculate Goal from R
        running_vals = 0
        G_t = []
        for rw in reversed(R):
            running_vals = running_vals * GAMMA + rw
            G_t.insert(0, running_vals)

        G_t = np.array(G_t)
        G_t -= G_t.mean()

        self.optimizer.zero_grad()
        state_tensor  = torch.FloatTensor(S)
        goal_tensor   = torch.FloatTensor(G_t)
        action_tensor = torch.LongTensor(A) # index of action

        log_probs = torch.log(self(state_tensor))
        select_probs = goal_tensor * log_probs[range(len(G_t)), action_tensor]
        loss = -select_probs.mean()
        loss.backward()
        self.optimizer.step()


        pass


def main():
    ENV_NAME    = 'CartPole-v0'
    NUM_EPISODE = 1000
    GAMMA       = 0.99


    episodes_reward = []
    # test

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape[0]
    n_action = env.action_space.n
    pg  = PG(state_shape, n_action)


    for episode_i in range(NUM_EPISODE):
        state = env.reset()
        total_reward = 0
        S = []
        A = []
        R = []

        while True:
            action_probs = pg.get_action(state)
            action       = np.random.choice(n_action, p = action_probs)

            next_state, reward, done, _ = env.step(action)

            S.append(state)
            A.append(action)
            R.append(reward)

            total_reward += reward

            state = next_state

            if done:
                print("State {}, reward : {}".format(episode_i, total_reward))
                episodes_reward.append(total_reward)
                pg.learn(S, A, R, GAMMA)
                break

    plt.plot(range(NUM_EPISODE), episodes_reward)
    plt.show()

if __name__ == '__main__':
    main()