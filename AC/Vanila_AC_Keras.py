import gym

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class Actor:
    def __init__(self, n_input, n_a, lr):
        self.n_input = n_input
        self.n_a     = n_a
        self.lr      = lr
        self.model   = self.create_model()
        self.grad_model = self.create_learn_model() # model only use for update parameters

    def custom_loss(self, Q):
        # L = sum {G * log(p(y | x)) } * Q
        def f(y_true, y_pred):
            outputs = K.sum(y_true * y_pred, axis=1)
            policy_loss = - K.sum(Q * K.log(outputs))
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
            optimizer=Adam(lr=self.lr),
            loss=self.custom_loss(goal_input)
        )

        return learn_model

class Critic:
    def __init__(self, n_input, n_a, lr):
        self.n_input = n_input
        self.n_a     = n_a
        self.lr      = lr
        self.Q       = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.n_input,))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        predictions = Dense(self.n_a, activation='linear')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(
            loss='mse',
            optimizer= Adam(lr=self.lr)
        )

        return model

def to_onehot(action, action_size):
    out = np.zeros(action_size)
    out[action] = 1
    return out

def to_numpy_arr(value):
    return np.expand_dims(value, axis=0)


def main():
    env = gym.make('CartPole-v0')

    n_inputs = env.observation_space.shape[0]
    n_a = env.action_space.n
    lr_a = 0.0005
    lr_c = 0.005

    GAMMA = 0.99
    NUM_EPISODE = 500

    actor  = Actor(n_inputs, n_a, lr_a)
    critic = Critic(n_inputs, n_a, lr_c)

    rewards = []
    for episode_i in range(NUM_EPISODE):
        state = env.reset()
        total_rw = 0
        while True:
            state_numpy = to_numpy_arr(state)
            action_prob = actor.model.predict(state_numpy)[0]
            action      = np.random.choice(n_a, p=action_prob)

            q_value_s   = critic.Q.predict(state_numpy)[0]
            q_s_at_a    = q_value_s[action]
            action_one_hot = to_onehot(action, n_a)

            # update actor
            actor.grad_model.train_on_batch([state_numpy, np.array([q_s_at_a]) ], np.array([action_one_hot]))

            next_s, rw, done, _ = env.step(action)
            # update critic
            next_state_numpy = to_numpy_arr(next_s)
            prob_a_next = actor.model.predict(next_state_numpy)[0]
            a_next      = np.random.choice(n_a, p=prob_a_next)
            q_target    = critic.Q.predict(next_state_numpy)[0][a_next]

            if not done:
                q_target = q_target * GAMMA + rw
            else:
                q_target = rw

            q_value_s[action] = q_target
            q_value_s = to_numpy_arr(q_value_s)

            critic.Q.train_on_batch(state_numpy, q_value_s)

            total_rw += rw
            state = next_s

            if done:
                print("Episode ", episode_i, " reward: ", total_rw)
                rewards.append(total_rw)
                break

    plt.plot(range(NUM_EPISODE), rewards)
    plt.show()




if __name__ == '__main__':
    main()