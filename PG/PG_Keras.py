import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

# reference :

class Policy:
    def __init__(self, n_input, n_a, lr):
        self.n_input = n_input
        self.n_a     = n_a
        self.lr      = lr
        self.model   = self.create_model()
        self.grad_model = self.create_learn_model() # model only use for update parameters

    def custom_loss(self, Goal):
        # L = sum {G * log(p(y | x)) }
        def f(y_true, y_pred):
            outputs = K.sum(y_true * y_pred, axis=1)
            policy_loss = - K.sum(Goal * K.log(outputs))
            return policy_loss
        return f

    def create_model(self):
        inputs = Input(shape=(self.n_input, ))
        x = Dense(32, activation='relu')(inputs)
        predictions = Dense(self.n_a, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        return model

    def create_learn_model(self):
        # Wrapper of policy model to update theta
        state = Input(shape=(self.n_input, ))
        goal_input = Input(shape=(1, ))
        x = self.model(state) # x: Tensor

        learn_model = Model([state, goal_input], x) # take state and goal as input, prob of action as output (one hot), loss = sum (goal * log(probs))
        learn_model.compile(
            optimizer='adam',
            loss=self.custom_loss(goal_input)
        )

        return learn_model

def to_onehot(action, action_size):
    out = np.zeros(action_size)
    out[action] = 1
    return out



def main():
    env = gym.make("CartPole-v0")

    n_inputs = env.observation_space.shape[0]
    n_a      = env.action_space.n
    lr       = 0.00025
    GAMMA    = 0.99
    pi = Policy(n_inputs, n_a, lr)


    NUM_EPISODE = 500

    rewards_history = []

    for episode_i in range(NUM_EPISODE):
        total_rw = 0
        state = env.reset()

        S = []
        A = []
        R = []

        while True:
            action_probs = pi.model.predict(np.expand_dims(state, axis=0))[0] # like predict list of state, with only 1 state, so we take the first [0] value
            action       = np.random.choice(n_a, p=action_probs)
            next_s, rw, done, _ = env.step(action)
            S.append(state)
            A.append(to_onehot(action, n_a))
            R.append(rw)
            total_rw += rw
            if done:
                break
            state = next_s

        # update policy parameter
        running_vals = 0
        G = []
        for rw in reversed(R):
            running_vals = running_vals * GAMMA + rw
            G.insert(0, running_vals)

        S = np.array(S)
        A = np.array(A)
        G = np.array(G)

        pi.grad_model.train_on_batch([S, G], A)

        rewards_history.append(total_rw)
        print("Episode #", episode_i, " total reward:", total_rw)

    plt.plot(range(NUM_EPISODE), rewards_history)
    plt.show()



if __name__ == '__main__':
    main()