# SOLVED CartPole after ~ 60 episodes

import gym
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *

import random
import matplotlib.pyplot as plt
from ER import ReplayBuffer
import time
class DeepQNetwork:
    def __init__(self, n_input, n_a, lr):
        self.n_input = n_input
        self.n_a     = n_a
        self.lr      = lr
        self.Q       = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.n_input,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.n_a, activation='linear')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(
            loss='mse',
            optimizer= Adam(lr=self.lr)
        )

        return model

def to_np(state):
    return np.array([state])

def main():
    env     = gym.make("CartPole-v0")
    n_a     = env.action_space.n
    n_input = env.observation_space.shape[0]
    lr      = 0.001
    DQN     = DeepQNetwork(n_input, n_a, lr)


    NUM_EPISODE = 100
    GAMMA       = 0.99
    DECAY_RATE  = 0.99
    EPS         = 1
    EPS_MIN     = 0.01
    BUFFER_MAX_SIZE = 2000
    BATCH_SIZE  = 32

    memory = ReplayBuffer(BUFFER_MAX_SIZE)

    reward_history = []

    start_time = time.time()
    for episode_i in range(NUM_EPISODE):
        state   = env.reset()
        total_rw = 0

        while True:
            state_np = to_np(state)
            Q_predict = DQN.Q.predict(state_np)[0]

            if random.random() > EPS:
                action    = Q_predict.argmax()

            else:
                action    = random.sample(range(n_a), 1)[0]
            EPS *= DECAY_RATE
            if EPS < EPS_MIN:
                EPS = EPS_MIN
            next_s, rw, done, _ = env.step(action)
            memory.add(state, action, rw, next_s, done)

            exps = memory.getBatch(BATCH_SIZE)
            states = np.array([e[0] for e in exps])
            actions = [e[1] for e in exps]
            rewards = [e[2] for e in exps]
            next_states = np.array([e[3] for e in exps])
            dones = [e[4] for e in exps]

            q_pred = DQN.Q.predict(states)
            q_next = DQN.Q.predict(next_states)


            for i in range(len(exps)):
                chosen_action = actions[i]
                q_pred[i][chosen_action] = rewards[i] + q_next[i].max() * (not dones[i]) * GAMMA


            DQN.Q.train_on_batch(states, q_pred)

            total_rw += rw
            state = next_s

            if done:
                reward_history.append(total_rw)
                print(episode_i, total_rw)
                break

    print(NUM_EPISODE,"episodes time :", time.time() - start_time)
    plt.plot(range(NUM_EPISODE), reward_history)
    plt.show()




if __name__ == '__main__':
    main()