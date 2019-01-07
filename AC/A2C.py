import gym

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# No experience replay
class Actor:
    def __init__(self, n_input, n_a, lr):
        self.n_input = n_input
        self.n_a     = n_a
        self.lr      = lr
        self.model   = self.create_model()
        self.grad_model = self.create_learn_model() # model only use for update parameters

    def custom_loss(self, td_error):
        # L = sum {td_error * log(p(action)) }
        def f(y_true, y_pred):
            outputs = K.sum(y_true * y_pred, axis=1)
            policy_loss = - K.sum(td_error * K.log(outputs))
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
        td_error = Input(shape=(1, ))
        action_probs = self.model(state) # x: Tensor

        learn_model = Model([state, td_error], action_probs) # take state and goal as input, prob of action as output (one hot), loss = sum (goal * log(probs))
        learn_model.compile(
            optimizer=Adam(lr=self.lr),
            loss=self.custom_loss(td_error)
        )

        return learn_model

class Critic:
    def __init__(self, n_input, n_out = 1, lr = 0.005):
        self.n_input = n_input
        self.n_out     = n_out
        self.lr      = lr
        self.V       = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.n_input,))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        predictions = Dense(self.n_out, activation='linear')(x)

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
    NUM_EPISODE = 250

    actor  = Actor(n_inputs, n_a, lr_a)
    critic = Critic(n_inputs, 1, lr_c)

    rewards = []
    for episode_i in range(NUM_EPISODE):
        state = env.reset()
        total_rw = 0
        while True:
            state_numpy = to_numpy_arr(state)
            action_prob = actor.model.predict(state_numpy)[0]
            action      = np.random.choice(n_a, p=action_prob)
            next_s, rw, done, _ = env.step(action)

            V_current   = critic.V.predict(state_numpy)[0]
            next_state_numpy = to_numpy_arr(next_s)
            V_next    = critic.V.predict(next_state_numpy)[0]

            target_V = V_next * (1-done) * GAMMA + rw
            td_error = target_V - V_current

            critic.V.train_on_batch(state_numpy, to_numpy_arr(target_V))

            action_one_hot = to_onehot(action, n_a)
            actor.grad_model.train_on_batch([state_numpy, np.array([td_error])], np.array([action_one_hot]))

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