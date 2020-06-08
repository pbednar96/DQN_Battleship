import numpy as np
from collections import deque
import random
import time
from copy import deepcopy

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from battleship2D_env import BattleShip2D

# ships have fix size SHIP_SIZEx1
BOARD_SIZE = 5
SHIP_SIZE = 3
SHIPS = 1
TOTAL_HIT = SHIP_SIZE * SHIPS

GAMMA = 0.85
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 64
LEARNING_RATE = 0.001

EPISODES = 2500

MODEL_NAME = f"{BOARD_SIZE ** 2}-128-128-{BOARD_SIZE ** 2}"

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = BattleShip2D(BOARD_SIZE, SHIPS, SHIP_SIZE)


class DQNAgent:

    # init class DQNAgent
    def __init__(self):
        self.env = BattleShip2D(BOARD_SIZE, SHIPS, SHIP_SIZE)
        self.state_size = self.env.board_size_2D
        self.action_size = self.env.board_size_2D
        # replay memory for store
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    # architecture of model
    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    # adds step data to replay memory
    # (old_state, action, reward, new_state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, state, step):
        # enough transition in replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([state[0] for state in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, env.board_size_2D))
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, env.board_size_2D))
        X = []
        y = []

        # for each in minibatch
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # get new q from future states
            if not done:
                max_future_q = np.max(future_qs_list[index])
                # bellman equation for new Q
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            # update Q value for given state
            current_qs = current_qs_list[index]
            action = action[0] * env.game_board_size + action[1]
            current_qs[action] = new_q

            # append train data
            X.append(current_state.reshape(-1, ))
            y.append(current_qs)

        # all data in batch
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, epochs=10, shuffle=False)

        if done:
            self.target_update_counter += 1

        # update target network with weights of main network
        if self.target_update_counter > 10:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # predict action by model
    def get_q(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


def save_reward_plot(log_stat):
    # plot with average reward during train
    plt.plot(np.arange(0, len(log_stat) * 50, 50), log_stat)
    plt.ylabel("Průměrná odměna")
    plt.xlabel("Episoda")
    plt.title(f"Model - {MODEL_NAME}")
    plt.savefig(f"logs/DQN_{EPISODES}_{env.game_board_size}x{env.game_board_size}_{MODEL_NAME}.png")
    plt.show()


def main():
    agent = DQNAgent()
    epsilon = 1
    start = time.time()
    tmp_reward = []
    stat_reward = []

    # for each episodes
    for episode in range(EPISODES):
        step = 1
        epsilon_reward = 0
        # reset environment and get initial state
        env = BattleShip2D(BOARD_SIZE, SHIPS, SHIP_SIZE)

        # information during train (unnecessary)
        if not episode % 100:
            end = time.time()
            print(episode)
            print(end - start)
            start = time.time()

        current_state = deepcopy(env.game_field)
        done = False
        while not done:

            if np.random.random() > epsilon:
                # get best action
                index_action = np.argmax(agent.get_q(current_state.reshape(-1, )))
                action = [int(index_action / BOARD_SIZE), index_action % BOARD_SIZE]
            else:
                action = env.sample()

            new_state, reward, done = env.step(action)
            epsilon_reward += reward
            # every step update replay memory and train
            new_state_copy = deepcopy(new_state)
            agent.update_replay_memory((current_state, action, reward, new_state_copy, done))
            agent.train(done, step)
            current_state = deepcopy(new_state)
            step += 1
        tmp_reward.append(epsilon_reward)

        # save average reward during train
        if not episode % 50:
            average_reward = sum(tmp_reward) / len(tmp_reward)
            stat_reward.append(average_reward)
            # print(f"Avg{average_reward}")
            tmp_reward = []

        # decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # save model to h5
    agent.model.save(f'DQN_{EPISODES}_{env.game_board_size}x{env.game_board_size}_{MODEL_NAME}.h5')
    save_reward_plot(stat_reward)


if __name__ == "__main__":
    main()
