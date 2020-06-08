import numpy as np


class BattleShip2D():
    # Create game board for battleship
    # States:
    # SHIP_DONE = 3
    # HIT = 2
    # MISS = 1
    # UNKNOWN = 0
    def __init__(self, board_size, number_ships, size_ship):
        self.game_board_size = board_size
        self.board_size_2D = board_size * board_size
        self.game_field = np.zeros((board_size, board_size))
        self.number_ships = number_ships
        self.size_ship = size_ship
        self.ship_tmp, self.ship_positions = self.generate_ship()
        self.hit = 0
        self.count_shot = 0
        self.total_hit = self.size_ship * self.number_ships
        self.done = False

    def generate_ship(self):
        # init ship on game board
        ships_tmp = []
        # 0 - vertical, 1 - horizontal
        direction_ship = np.random.randint(0, 2, self.number_ships)
        ship_position = []
        same_position = True
        # set ship positions
        while same_position:
            for j in direction_ship:
                tmp = []
                if j == 0:
                    ship_random = np.random.randint(self.game_board_size - self.size_ship + 1)
                    ship_column = np.random.randint(self.game_board_size)
                    for index in range(self.size_ship):
                        ship_position.append([ship_random + index, ship_column])
                        tmp.append([ship_random + index, ship_column])
                if j == 1:
                    ship_random = np.random.randint(self.game_board_size - self.size_ship + 1)
                    ship_row = np.random.randint(self.game_board_size)
                    for index in range(self.size_ship):
                        ship_position.append([ship_row, ship_random + index])
                        tmp.append([ship_row, ship_random + index])
                ships_tmp.append(tmp)
            if len(np.unique(ship_position, axis=0)) == len(ship_position):
                same_position = False
            else:
                ship_position = []
        return ships_tmp, ship_position

    def step(self, action):
        # use action in game
        reward = -0.1
        self.count_shot += 1
        if self.game_field[action[0]][action[1]] != 0:
            reward -= self.game_board_size
        else:
            for item_ship in self.ship_positions:
                if np.array_equal(action, item_ship):
                    self.game_field[action[0]][action[1]] = 2
                    self.hit += 1
                    reward += 0.2
                    break
                else:
                    self.game_field[action[0]][action[1]] = 1
                    # reward += -0.1

            # check if ship is done
            for _ in range(self.board_size_2D):
                for x_ship in self.ship_tmp:
                    done = True
                    for k in x_ship:
                        if self.game_field[k[0]][k[1]] != 2:
                            done = False
                    if done:
                        for k in x_ship:
                            self.game_field[k[0]][k[1]] = 3
                        reward += self.game_board_size * 0.8

            if self.hit == self.total_hit:
                self.done = True
        if self.count_shot >= self.board_size_2D:
            self.done = True

        return self.game_field, reward, self.done

    def sample(self):
        # random action in game field
        x = np.random.randint(self.game_board_size)
        y = np.random.randint(self.game_board_size)
        action = [x, y]

        return action
