import numpy as np
import random
from hashlib import md5
from datetime import datetime
from enum import Enum
import copy

EMPTY = -1
RED = 0
YELLOW = 1

NUM_OF_ROWS = 6
NUM_OF_COLUMNS = 7

ALPHA = 0.05
GAMMA = 0.95


class InsertStatus(Enum):
    won = 1
    prevented = 2
    full = 3
    success = 4
    draw = 5


class Bot(object):
    def __init__(self):
        self.state_str_to_id = {}
        self.actions = range(NUM_OF_COLUMNS)
        self.q = {RED: EfficientNPArray(), YELLOW: EfficientNPArray()}
        self.wins_count = {"computer": 0, "opponent": 0}
        self.num_of_states = 0
        self.num_of_games_played = 0
        self.games = 1000000
        self.epsilon = 0.5
        self.min_epsilon = 0.05

    def get_or_create_state(self, board):
        #         now = datetime.now()
        raw_state = md5(str(board)).digest()
        state = self.state_str_to_id.get(raw_state)
        if state is None:
            state = self.num_of_states
            self.state_str_to_id[raw_state] = state
            self.q[RED].add()
            self.q[YELLOW].add()
            self.num_of_states += 1
        return state

    def train_bot(self, self_training=False, computer2=None, smart_random=False, winning_threshold=0.8,
                  time_threshold=None, generate_random_states=False):
        computer_color = RED
        last_updated = now = datetime.now()
        a = ConnectFour()
        old_epsilon = self.epsilon
        if self_training:
            computer2 = self
        while not time_threshold or (datetime.now() - now).total_seconds() < time_threshold:
            if self.num_of_games_played and self.num_of_games_played % 1000 == 0:
                state = self.get_or_create_state(a.board)
                print self.q[RED].data[state]
                print "Epsilon was reduced: [old_value: %s, new_value: %s]" % (old_epsilon, self.epsilon)
                print "Number of states visited so far: %s" % self.num_of_states
                old_epsilon = self.epsilon
                print self.num_of_games_played, (datetime.now() - last_updated).total_seconds()
                last_updated = datetime.now()
                print self.wins_count
                if self.wins_count["computer"] > winning_threshold * \
                        (self.wins_count["computer"] + self.wins_count["opponent"]):
                    break
                self.wins_count = {"computer": 0, "opponent": 0}
            self.play_game(computer_color, computer2=computer2, smart_random=smart_random,
                           generate_random_state=generate_random_states)
            if self.epsilon > self.min_epsilon:
                self.epsilon -= 1.0 / self.games
            self.num_of_games_played += 1
            computer_color = YELLOW if computer_color == RED else RED

    def play_game(self, computer_color, computer2=None, is_user=False, smart_random=False,
                  generate_random_state=False):
        g = ConnectFour(generate_random_state=generate_random_state)
        opponent_color = 1 - computer_color
        state = self.get_or_create_state(g.board)
        still_playing = True
        turn = RED
        computer2_state = None
        previous_state = None
        previous_action = None
        previous_status = None
        while still_playing:
            # col = input('{}\'s turn: '.format('Red' if turn == RED else 'Yellow'))
            if is_user:
                g.print_board()
            if computer2 and turn == opponent_color:
                computer2_state = computer2.get_or_create_state(g.board)
            action = self.choose_action(g, computer_color, state) if turn == computer_color \
                else input('action: ') if is_user \
                else computer2.choose_action(g, opponent_color, computer2_state, is_training_bot=True) if computer2 \
                else g.check_for_best_possibly_move(turn) if smart_random \
                else random.randrange(NUM_OF_COLUMNS)

            status = g.insert(action, turn)
            still_playing = status not in {InsertStatus.won, InsertStatus.draw}
            if status == InsertStatus.won:
                w = "computer" if computer_color == turn else "opponent"
                self.wins_count[w] += 1
            next_state = self.get_or_create_state(g.board)
            if previous_state is not None and turn == computer_color:
                self.update_q(previous_status, turn, computer_color, previous_state, state, previous_action)
            if status == InsertStatus.won and turn == opponent_color:
                self.update_q(status, turn, computer_color, previous_state, next_state, previous_action)
            if status == InsertStatus.won and turn == computer_color:
                self.update_q(status, turn, computer_color, state, next_state, action)

            if turn == computer_color:
                previous_state = state
                previous_action = action
                previous_status = status

            if status in {InsertStatus.success, InsertStatus.prevented}:
                state = next_state
                turn = YELLOW if turn == RED else RED
            # if turn != computer_color and status == InsertStatus.won:
            #     g.print_board()
            #     pass

    def get_new_q_regard_rsa(self, computer_rsa, computer_color, state, next_state, action):
        computer_qsa = self.q[computer_color].data[state, action]
        computer_new_q = computer_qsa + ALPHA * (computer_rsa + GAMMA * self.q[computer_color].data[next_state].max()
                                                 - computer_qsa)

        return computer_new_q

    def update_q(self, status, turn, computer_color, previous_state, state, action):
        if status == InsertStatus.won:
            rsa = 50 if computer_color == turn else -50
            computer_new_q = self.get_new_q_regard_rsa(rsa, computer_color, previous_state, state, action)
        else:
            computer_new_q = self.get_new_q_regard_rsa(-1, computer_color, previous_state, state, action)

        self.q[computer_color].data[previous_state, action] = computer_new_q

    def get_q(self, computer_color, state, a):
        return self.q[computer_color].data[state, a] if state is not None else 0

    def choose_action(self, g, color, state, is_training_bot=False):
        q = [self.get_q(color, state, a) if g.is_valid_move(a) else -float("inf") for a in self.actions]
        max_q = max(q)

        action = None
        if is_training_bot:
            action = g.check_for_best_possibly_move(color, random_if_not_found=False)

        if action is None:
            if not is_training_bot and random.random() < self.epsilon:
                valid_q = [i for i in xrange(len(q)) if q[i] != -float("inf")]
                i = random.choice(valid_q)
            else:
                best = [i for i in xrange(len(q)) if q[i] == max_q]
                i = best[len(best) // 2]

            action = self.actions[i]
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
        return copy.deepcopy(self.board)

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

    def check_for_best_possibly_move(self, color, random_if_not_found=True):
        prevented_columns = list()
        for column in xrange(NUM_OF_COLUMNS):
            status = self.insert(column, color, dry_insert=True)
            if status == InsertStatus.prevented:
                prevented_columns.append(column)
            elif status == InsertStatus.won:
                return column
        return random.choice(prevented_columns) if prevented_columns \
            else random.randrange(NUM_OF_COLUMNS) if random_if_not_found \
            else None

    def is_valid_move(self, column):
        return self.board[0][column] == EMPTY

    @staticmethod
    def get_empty_board():
        return [[EMPTY] * NUM_OF_COLUMNS for _ in xrange(NUM_OF_ROWS)]

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


class EfficientNPArray(object):
    def __init__(self):
        self.capacity = 100
        self.data = np.zeros(shape=(self.capacity, NUM_OF_COLUMNS))
        self.size = 0

    def add(self):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros(shape=(self.capacity, NUM_OF_COLUMNS))
            newdata[:self.size] = self.data
            self.data = newdata

        self.size += 1


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


if __name__ == '__main__':
    bot = Bot()
    bot.train_bot()
