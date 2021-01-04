import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_Name = "CartPole-v1"
HIDDEN_LAYER_SIZE = 24

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

LEARNING_RATE = 0.001
GAMMA = 0.95


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.action_space = action_space

        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)  # deque for storing every step

        self.model = Sequential()
        self.model.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(observation_space,), activation="relu"))  # first hidden layer with 4 inputs which is the observation space
        self.model.add(Dense(HIDDEN_LAYER_SIZE, activation="relu"))  # Second hidden layer

        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)  # get a random action
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:  # don't replay if the number of steps is less than batch
            return
        batch = random.sample(self.memory, BATCH_SIZE)  # get a random set of iterations of length BATCH_SIZE
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY  # slowly narrow the rate
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)  # Ensure rate doesn't go below the minimum

def cartpole():
    env = gym.make("CartPole-v1")  # get the pre-made environment
    """
    Observation Space: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
    """
    observation_space = env.observation_space.shape[0]  # Number of possible state values
    # print("Observation Space: ", observation_space)
    """
    Action Space:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
    """
    action_space = env.action_space.n  # Number of actions that can be taken
    # print("Action Space: ", action_space)
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        # Every episode, make a new environment and reset the initial state
        state = env.reset()  # set the initial state to 4 random numbers
        state = np.reshape(state, [1, observation_space])

        step = 0
        while True:
            step += 1
            env.render()  # render a new environment for the current episode
            action = dqn_solver.act(state)  # get the next action from the DQN solver. Pass through the current state
            state_next, reward, terminal, info = env.step(action)  # step along with that action and get the next state and the reward
            reward = reward if not terminal else -reward  # if the pole has gone out of bounds, make the reward negative
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)  # remember everything
            dqn_solver.experience_replay()
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
            dqn_solver.experience_replay()
        # env.close()

if __name__ == "__main__":
    cartpole()