import gym
import numpy as np
import time

#https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
# _init_ conatain information

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
GAMMA = 0.95
EPISODES = 25000

SHOW_EVERY = 100

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# print(DISCRETE_OS_WIN_SIZE)

q_table = np.random.uniform(low=0, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    return tuple(discrete_state.astype(np.int))

def render_env(episode):
    if episode % SHOW_EVERY == 0:
        return True
    else:
        return False


def q_learning_mountain_car():
    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)
            print(new_discrete_state)
            # print(new_discrete_state)
            if render_env(episode):
                time.sleep(0.003)
                env.render()
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                            reward + GAMMA * max_future_q)
                q_table[discrete_state + (action,)] = new_q

            elif new_state[0] >= env.goal_position:
                print(f"We made it on episode: {episode}")
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

    env.close()

def main():
    print(DISCRETE_OS_SIZE)
    q_learning_mountain_car()


if __name__ == "__main__":
    main()