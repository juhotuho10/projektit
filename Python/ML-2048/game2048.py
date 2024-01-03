import numpy as np
import random
from numba import jit

@jit(nopython=True)
def compress(grid: np.ndarray) -> np.ndarray:
    # Compresses the grid by shifting all non-zero elements to the left.
    new_grid = np.zeros_like(grid)
    for i in range(4):
        non_zero_elements = grid[i][grid[i] != 0]
        new_grid[i, :len(non_zero_elements)] = non_zero_elements
    return new_grid

@jit(nopython=True)
def merge(grid:np.ndarray) -> np.ndarray:
    # Merges adjacent identical non-zero elements in each row by doubling their value.
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j + 1] and grid[i, j] != 0:
                grid[i, j] *= 2
                grid[i, j + 1] = 0
    return grid

@jit(nopython=True)
def reverse(grid: np.ndarray) -> np.ndarray:
    # Reverses each row of the grid. Used for 'right' move.
    reversed_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        reversed_grid[i] = grid[i][::-1]
    return reversed_grid

@jit(nopython=True)
def transpose(grid: np.ndarray) -> np.ndarray:
    # Transposes the grid. Used for 'up' and 'down' moves.
    return np.transpose(grid)

class Game2048:
    def __init__(self, size=4):
        # Initializes the game with a given size.
        self.size = size
        self.seed = 42
        self.reset()

    def reset(self):
        # Resets the game board to the initial state.
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        # Adds a new tile (2 or 4) at a random empty position on the board.
        zero_coords = np.where(self.board == 0)
        zero_coord_tuples = list(zip(zero_coords[0], zero_coords[1]))
        if zero_coord_tuples:
            i, j = random.choice(zero_coord_tuples)
            self.board[i, j] = random.choice([2, 4])

    def move(self, direction):
        # Executes a move in the given direction, applying compression, merging, and reversal/transposition as needed.
        match direction:
            case "left":
                self.board = compress(self.board)
                self.board = merge(self.board)
                self.board = compress(self.board)
            case "right":
                self.board = reverse(self.board)
                self.move("left")
                self.board = reverse(self.board)
            case "up":
                self.board = transpose(self.board)
                self.move("left")
                self.board = transpose(self.board)
            case "down":
                self.board = transpose(self.board)
                self.move("right")
                self.board = transpose(self.board)

    def move_board(self, direction):
        # Executes a move and then adds a new tile to the board.
        assert direction in ["left", "right", "up", "down"]
        self.move(direction)
        self.add_new_tile()

    def board_from_move(self, direction):
        # Returns the board state after making a move in the given direction.
        temp_game = Game2048()
        temp_game.board = np.copy(self.get_board())
        temp_game.move_board(direction)
        return temp_game.get_board()
    
    def get_board(self):
        # Returns the current board state.
        return self.board
