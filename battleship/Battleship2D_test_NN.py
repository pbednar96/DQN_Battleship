import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, mode
from collections import Counter

import time

from keras.models import load_model

from battleship.battleship2D_env import BattleShip2D

BOARD_SIZE = 5
SHIP_SIZE = 3
SHIPS = 1
TOTAL_HIT = SHIP_SIZE * SHIPS
# ships have fix size 3x1

MODEL_NAME = "25-128-128-25"
SHOW_GAME = True


def show_plots(game_field, probability,i, iter):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(game_field, cmap="hot", vmin=0, vmax=3)
    ax2.imshow(probability.reshape(BOARD_SIZE, BOARD_SIZE), cmap="RdBu_r", vmin=-5, vmax=5)
    plt.savefig(f"/home/petr/Pictures/battleship/{i}_{iter}.png")
    # plt.show()

def show_hist_freq(num_shots):
    freq_shot = Counter(sorted(num_shots))
    print(freq_shot)
    plt.bar(range(len(freq_shot)), list(freq_shot.values()), align='center')
    plt.xticks(range(len(freq_shot)), list(freq_shot.keys()))
    plt.xlabel("Počet střel")
    plt.ylabel("Počet her")
    plt.title(f"Model - {MODEL_NAME}")

    plt.show()


def main():
    model = load_model('DQN_7500_5x5_25-128-128-25.h5')

    while True:
        num_shot = []
        reward_game = 0
        for game in range(50):
            env = BattleShip2D(BOARD_SIZE, SHIPS, SHIP_SIZE)
            action_log = []
            reward = 0
            iteration = 0
            while not env.done:
                iteration += 1
                # print(f"Game:\n {env.game_field}")
                model_result = model.predict(env.game_field.reshape(1, env.board_size_2D))
                # next action not allow action used in history
                # for i in action_log:
                #     model_result[0][i] = min(model_result[0]) - 0.1
                # print(f"NN result: \n {model_result.reshape(env.game_board_size, env.game_board_size)}")
                action = np.argmax(model_result)
                action_log.append(action)
                action = [int(action / env.game_board_size), action % env.game_board_size]
                if SHOW_GAME:
                    show_plots(env.game_field, model_result, game, iteration)
                _, r, _ = env.step(action)
                reward += r
            reward_game += reward
            num_shot.append(iteration)

            if SHOW_GAME:
                show_plots(env.game_field, model_result, game, "final")
                print(f"End:\n {env.game_field}")
        print(f"Avrage: {mean(num_shot)}")
        if mean(num_shot) > 9.5:
            break
    print(reward_game/50)
    print(f"Avrage: {mean(num_shot)}")
    print(f"Median: {median(num_shot)}")
    # print(f"Modus: {mode(num_shot)}")
    # plt.hist(num_shot, bins=max(num_shot) - min(num_shot))
    # print(num_shot)
    # plt.xlabel("Počet střel")
    # plt.ylabel("Četnost")
    # plt.show()
    show_hist_freq(num_shot)


if __name__ == "__main__":
    main()
