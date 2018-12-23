import gym
import numpy as np
import matplotlib.pyplot as plt


class Policy:
    def __init__(self, num_input, n_a, lr):
        self.w = np.random.rand(num_input, n_a) # (4, 2)
        self.lr = lr

    def forward(self, state):
        # state: shape (4, )
        z = state.dot(self.w)
        exp = np.exp(z) # softmax
        return exp / np.sum(exp)


    def backward(self, state, probs, action):
        probs[action] = probs[action] - 1 # d_softmax
        probs /= probs[action] # dlog
        dW = state[:, None].dot( probs[None, :] ) # 4x1 x 1x2 => 4x2
        return dW

    def update(self, grads):
        self.w += grads * self.lr


def main():
    env = gym.make("CartPole-v0")
    num_input = env.observation_space.shape[0]
    n_a = env.action_space.n

    NUM_EPISODES = 3000
    GAMMA = 0.99
    pi = Policy(num_input, n_a, 0.000025)
    rws = []
    for episode_i in range(NUM_EPISODES):
        state = env.reset()
        total_rw = 0
        R = []
        grads = []

        while True:
            probs = pi.forward(state)
            action = np.random.choice(n_a, p=probs)

            next_s, rw, done, _ = env.step(action)

            grad = pi.backward(state, probs, action)

            R.append(rw)
            grads.append(grad)
            total_rw += rw
            state = next_s

            if done:
                print("Episode ", episode_i, " rw: ", total_rw)
                break

        # calculate goal
        running_vals = 0
        G_t = []
        for rw in reversed(R):
            running_vals = running_vals * GAMMA + rw
            G_t.insert(0, running_vals)

        # update weight
        for grad, G in zip(grads, G_t):
            pi.update(grad * G)
        rws.append(total_rw)

    plt.plot(range(NUM_EPISODES), rws)
    plt.show()
    ## Uncomment to see agent play
    # while True:
    #     done = False
    #     state = env.reset()
    #     total_rw = 0
    #     while not done:
    #         env.render()
    #         probs = pi.forward(state)
    #         action = probs.argmax()
    #         next_s, rw, done, _ = env.step(action)
    #         state = next_s
    #         total_rw += rw
    #         if done:
    #             print("die, rw = ", total_rw)

if __name__ == '__main__':
    main()