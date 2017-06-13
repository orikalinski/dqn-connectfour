import numpy as np
import random
from datetime import datetime
import h5py

from enum import Enum
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import RMSprop

EMPTY = 0
RED = 1
YELLOW = 2

NUM_OF_ROWS = 6
NUM_OF_COLUMNS = 7

InsertStatus = Enum("InsertStatus", "won prevented full draw success")

GAMMA = 0.9
ALPHA = 0.05


class Bot(object):
    def __init__(self):
        self.shape = NUM_OF_ROWS * NUM_OF_COLUMNS * 2
        self.wins_count = {"computer": 0, "opponent": 0}
        self.num_of_games_played = 0
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epochs = 40000
        self.models_dict = {RED: self.initialize_model(), YELLOW: self.initialize_model()}
        self.X_train = {RED: [], YELLOW: []}
        self.Y_train = {RED: [], YELLOW: []}

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(1024, kernel_initializer='lecun_uniform', input_shape=(self.shape,)))
        model.add(Activation(K.relu))
        model.add(Dropout(0.2))  # I'm not using dropout, but maybe you wanna give it a try?
        #
        # model.add(Dense(512, kernel_initializer='lecun_uniform'))
        # model.add(Activation(K.relu))
        # model.add(Dropout(0.2))
        # #
        # model.add(Dense(256))
        # model.add(Activation(K.relu))
        # model.add(Dropout(0.2))

        model.add(Dense(7, kernel_initializer='lecun_uniform', activation='linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)
        return model

    def train_model(self, training_bots=None, smart_random=False, winning_threshold=0.8, time_threshold=None,
                    generate_random_states=False):
        computer_color = RED
        last_updated = now = datetime.now()
        a = ConnectFour()

        training_bot = None
        epochs = {RED: 0, YELLOW: 0}
        while not time_threshold or (datetime.now() - now).total_seconds() < time_threshold:
            if self.num_of_games_played and self.num_of_games_played % 1000 == 0:
                state = a.convert_to_three_dimensional_array()
                print self.models_dict[RED].predict(state.reshape(1, self.shape), batch_size=1)
                print self.num_of_games_played, (datetime.now() - last_updated).total_seconds()
                last_updated = datetime.now()
                print self.wins_count
                if self.wins_count["computer"] > winning_threshold * \
                        (self.wins_count["computer"] + self.wins_count["opponent"]):
                    break
                self.wins_count = {"computer": 0, "opponent": 0}
            if training_bots:
                training_bot = random.choice(training_bots)
            self.play_game(computer_color, computer2=training_bot, smart_random=smart_random,
                           generate_random_state=generate_random_states)
            epochs[computer_color] += 1
            if self.epsilon > self.min_epsilon:
                self.epsilon -= 1.0 / self.epochs
            self.num_of_games_played += 1
            # computer_color = YELLOW if computer_color == RED else RED
            if self.num_of_games_played and self.num_of_games_played % 20 == 0:
                self.fit_model(self.X_train, self.Y_train, epochs)
                self.X_train = {RED: [], YELLOW: []}
                self.Y_train = {RED: [], YELLOW: []}
                epochs = {RED: 0, YELLOW: 0}

        if epochs[YELLOW] or epochs[RED]:
            self.fit_model(self.X_train, self.Y_train, epochs)
            self.X_train = {RED: [], YELLOW: []}
            self.Y_train = {RED: [], YELLOW: []}

    def play_game(self, computer_color, computer2=None, is_user=False, smart_random=False,
                  generate_random_state=False):
        g = ConnectFour(generate_random_state=generate_random_state)
        opponent_color = YELLOW if computer_color == RED else RED
        state = g.convert_to_three_dimensional_array()
        still_playing = True
        turn = RED
        computer2_state = None
        previous_state = None
        previous_action = None
        previous_status = None
        while still_playing:
            if is_user:
                g.print_board()
            if computer2:
                computer2_state = g.convert_to_three_dimensional_array()
            action = self.choose_action(g, computer_color, state) if turn == computer_color \
                else input('action: ') if is_user \
                else computer2.choose_action(g, opponent_color, computer2_state, should_use_epsilon=False) if computer2 \
                else g.check_for_best_possibly_move(turn) if smart_random \
                else random.randrange(NUM_OF_COLUMNS)

            status = g.insert(action, turn)
            still_playing = status not in {InsertStatus.won, InsertStatus.draw}
            if status == InsertStatus.won:
                w = "computer" if computer_color == turn else "opponent"
                self.wins_count[w] += 1
            next_state = g.convert_to_three_dimensional_array()
            if previous_state is not None and turn == computer_color:
                x, y = self.update_q(previous_status, turn, computer_color, previous_state, state,
                                     previous_action)
                self.X_train[computer_color].append(x)
                self.Y_train[computer_color].append(y)
            if previous_state is not None and not still_playing and turn == opponent_color:
                x, y = self.update_q(status, turn, computer_color, previous_state, next_state, previous_action)
                self.X_train[computer_color].append(x)
                self.Y_train[computer_color].append(y)
            elif not still_playing and turn == computer_color:
                x, y = self.update_q(status, turn, computer_color, state, next_state, action)
                self.X_train[computer_color].append(x)
                self.Y_train[computer_color].append(y)

            if turn == computer_color:
                previous_state = state.copy()
                previous_action = action
                previous_status = status

            if status in {InsertStatus.success, InsertStatus.prevented}:
                state = next_state.copy()
                turn = YELLOW if turn == RED else RED
            # if turn != computer_color and status == InsertStatus.won:
            #     g.print_board()

    def get_new_q_regard_rsa(self, rsa, computer_color, next_state):
        next_state_qsa = self.models_dict[computer_color]\
            .predict(next_state.reshape(1, self.shape), batch_size=1)
        new_q = rsa + GAMMA * np.max(next_state_qsa)
        return new_q

    def update_q(self, status, turn, computer_color, previous_state, state, action):
        x = previous_state.reshape(1, self.shape)
        q = self.models_dict[computer_color].predict(x, batch_size=1)
        if status == InsertStatus.won:
            rsa = 100 if computer_color == turn else -100
            new_q = rsa
        elif status == InsertStatus.draw:
            rsa = -10
            new_q = rsa
        else:
            rsa = -1
            new_q = self.get_new_q_regard_rsa(rsa, computer_color, state)

        y = np.zeros((1, NUM_OF_COLUMNS))
        y[:] = q[:]
        y[0][action] = new_q
        return x[0], y[0]

    def fit_model(self, X_train, Y_train, epochs):
        for color in [RED, YELLOW]:
            if X_train[color]:
                X_train_color = np.array(X_train[color])
                Y_train_color = np.array(Y_train[color])
                self.models_dict[color].fit(X_train_color, Y_train_color,
                                            batch_size=len(X_train[color]), epochs=epochs[color], verbose=0)

    def choose_action(self, g, computer_color, state, should_use_epsilon=True):
        q = self.models_dict[computer_color].predict(
            state.reshape(1, self.shape), batch_size=1)[0]
        q = [v if g.is_valid_move(action) else -float("inf") for action, v in enumerate(q)]

        if should_use_epsilon and random.random() < self.epsilon:  # choose random action
            valid_q = [i for i in xrange(len(q)) if q[i] != -float("inf")]
            action = random.choice(valid_q)
        else:  # choose best action from Q(s,a) values
            action = np.argmax(q)

        return action


class ConnectFour(object):
    def __init__(self, generate_random_state=False):
        """Create a new game."""
        if generate_random_state:
            self.board, self.num_of_moves_played = self.get_random_board()
        else:
            self.board = self.get_empty_board()
            self.num_of_moves_played = 0

    def get_copied_board(self):
        return self.board.copy()

    def insert(self, column, color, insert_only=False, index_only=False, dry_insert=False, board=None):
        """Insert the color in the given column."""
        changed_color = False
        nrow = NUM_OF_ROWS - 1
        board = self.board if board is None else board
        for i, row in enumerate(board[::-1]):
            col = row[column]
            if col == EMPTY:
                changed_color = True
                nrow -= i
                break

        if index_only:
            return nrow

        if not changed_color:
            return InsertStatus.full

        return_status = InsertStatus.success
        if not insert_only:
            board[nrow][column] = YELLOW if color == RED else RED
            prevented = self.get_winner() is not None
        board[nrow][column] = color
        if insert_only:
            return return_status

        if prevented:
            return_status = InsertStatus.prevented
        else:
            won = self.get_winner() is not None
            if won:
                return_status = InsertStatus.won

        if dry_insert:
            board[nrow][column] = EMPTY
        else:
            self.num_of_moves_played += 1
            if return_status != InsertStatus.won and self.num_of_moves_played >= NUM_OF_COLUMNS * NUM_OF_ROWS:
                return InsertStatus.draw

        return return_status

    def check_for_best_possibly_move(self, color):
        prevented_columns = list()
        for column in xrange(NUM_OF_COLUMNS):
            status = self.insert(column, color, dry_insert=True)
            if status == InsertStatus.prevented:
                prevented_columns.append(column)
            elif status == InsertStatus.won:
                return column
        return random.choice(prevented_columns) if prevented_columns else random.randrange(NUM_OF_COLUMNS)

    def is_valid_move(self, column):
        return self.board[0][column] == EMPTY

    def convert_to_three_dimensional_array(self):
        new_board = np.zeros(shape=(2, NUM_OF_ROWS, NUM_OF_COLUMNS))
        new_board[RED - 1] = (self.board == RED).astype(float)
        new_board[YELLOW - 1] = (self.board == YELLOW).astype(float)
        return new_board

    @staticmethod
    def get_empty_board():
        return np.zeros(shape=(NUM_OF_ROWS, NUM_OF_COLUMNS))

    def get_random_board(self):
        num_of_moves_to_make = random.choice(range(0, 15, 2))
        num_of_moves_played = 0
        valid_board = False
        while not valid_board:
            board = self.get_empty_board()
            turn = RED
            while num_of_moves_played < num_of_moves_to_make:
                column = random.randrange(NUM_OF_COLUMNS)
                status = self.insert(column, turn, insert_only=True, board=board)
                if status == InsertStatus.success:
                    num_of_moves_played += 1
                turn = YELLOW if turn == RED else RED
            valid_board = self.get_winner(board=board) is None
        return board, num_of_moves_to_make

    def get_winner(self, board=None):
        """Get the winner on the current board."""

        board = self.board if board is None else board

        # check horizontal spaces
        for y in xrange(NUM_OF_COLUMNS):
            for x in xrange(NUM_OF_ROWS - 3):
                if EMPTY != board[x][y] == board[x + 1][y] == board[x + 2][y] == board[x + 3][y]:
                    return board[x][y]

        # check vertical spaces
        for x in xrange(NUM_OF_ROWS):
            for y in xrange(NUM_OF_COLUMNS - 3):
                if EMPTY != board[x][y] == board[x][y + 1] == board[x][y + 2] == board[x][y + 3]:
                    return board[x][y]

        # check / diagonal spaces
        for x in xrange(NUM_OF_ROWS - 3):
            for y in xrange(3, NUM_OF_COLUMNS):
                if EMPTY != board[x][y] == board[x + 1][y - 1] == board[x + 2][y - 2] == \
                        board[x + 3][y - 3]:
                    return board[x][y]

        # check \ diagonal spaces
        for x in xrange(NUM_OF_ROWS - 3):
            for y in xrange(NUM_OF_COLUMNS - 3):
                if EMPTY != board[x][y] == board[x + 1][y + 1] == board[x + 2][y + 2] == \
                        board[x + 3][y + 3]:
                    return board[x][y]

    def print_board(self):
        """Print the board."""
        print('   '.join(map(str, range(NUM_OF_COLUMNS))))
        for row in self.board:
            print('  '.join(str(col).zfill(2) for col in row))


def run_vs_user(computer):
    last_updated = datetime.now()
    num_of_games = 0
    computer_color = RED
    while True:
        if num_of_games and num_of_games % 5 == 0:
            print num_of_games, (datetime.now() - last_updated).total_seconds()
            last_updated = datetime.now()
            print computer.wins_count
            computer.wins_count = {"computer": 0, "opponent": 0}
        computer.play_game(computer_color, is_user=True)
        computer_color = YELLOW if computer_color == RED else RED
        num_of_games += 1


def main():
    bot = Bot()
    bot.train_model(generate_random_states=True, winning_threshold=0.98)
    # bot.epsilon = 0
    # bot.models_dict[RED] = load_model("/tmp/model_red.h5")
    # while True:
    #     bot.play_game(RED, is_user=True)


if __name__ == '__main__':
    main()
