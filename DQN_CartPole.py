import gym
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

LEARNING_RATE = .001
SIZE_DATASET = 3000

#another env need couple of reworks
# - input layer dynamic
# - output layer dynamic


env = gym.make("CartPole-v0")
env.reset()


def create_dateset():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(SIZE_DATASET):
        env.reset()
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(200):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= 40:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])

        scores.append(score)

    return training_data


def create_model():
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model


def train_model(training_data, model=False):
    observation_state = [i[0] for i in training_data]
    target = [i[1] for i in training_data]
    model = create_model()
    for state, target_f in zip(observation_state, target):
        state = np.array(state).reshape(1, 4)
        target_f = np.array(target_f).reshape(1, 2)
        model.fit(state, target_f, epochs=5, verbose=0)
    return model


def main():
    train_data = create_dateset()
    print("Dataset was created!")
    model = train_model(train_data)
    print("Training done!")
    for i in range(10):
        env.reset()
        env.render()
        score = 0
        prev_observation = []
        for _ in range(200):
            env.render()
            time.sleep(0.03)

            if len(prev_observation) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_observation.reshape(1, 4))[0])
            new_observation, reward, done, info = env.step(action)
            prev_observation = new_observation
            # print(prev_observation.reshape(1, 4))

            score += reward
            if done:
                break

        print(f"{i}: score:{score}")


if __name__ == "__main__":
    main()
