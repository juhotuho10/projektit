import numpy as np
import random

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.seed = 42
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()


    
    def add_new_tile(self):

        new_tile = []

        zero_coords = np.where(self.board == 0)
        zero_coord_tuples = list(zip(zero_coords[0], zero_coords[1]))
        if zero_coord_tuples:
            i, j = random.choice(zero_coord_tuples)
            num_coice = random.choice([2, 4])
            self.board[i, j] = num_coice
            new_tile = [[i, j], num_coice]

        return new_tile
    
    
    def set_new_tile(self, tile):
        if tile:
            i, j = tile[0]
            number = tile[1]
            self.board[i, j] = number

    
    def compress(self, grid: np.ndarray):

        new_grid = np.zeros_like(grid)
        for i in range(4):
            non_zero_elements = grid[i][grid[i] != 0]
            new_grid[i, :len(non_zero_elements)] = non_zero_elements
        return new_grid

    def merge(self, grid:np.ndarray):
        for i in range(4):
            for j in range(3):
                if grid[i, j] == grid[i, j + 1] and grid[i, j] != 0:
                    grid[i, j] *= 2
                    grid[i, j + 1] = 0
        return grid

    def reverse(self, grid: np.ndarray):
        return np.flip(grid, axis=1)

    def transpose(self, grid: np.ndarray):
        return np.transpose(grid)
    
    def move(self, direction):

        match direction:
            case "left":
                self.board = self.compress(self.board)
                self.board = self.merge(self.board)
                self.board = self.compress(self.board)

            case "right":
                self.board = self.reverse(self.board)
                self.move("left")
                self.board = self.reverse(self.board)

            case "up":
                self.board = self.transpose(self.board)
                self.move("left")
                self.board = self.transpose(self.board)

            case "down":
                self.board = self.transpose(self.board)
                self.move("right")
                self.board = self.transpose(self.board)


    def move_board(self, direction, set_tile):
        assert direction in ["left", "right", "up", "down"]

        self.move(direction)
        self.set_new_tile(set_tile)

    def fake_move(self, direction):

        self.move(direction)
        new_tile = self.add_new_tile()
        return new_tile
    
        
    def is_game_over(self):
        if np.any(self.board == 0):
            return
        for i in range(self.size):
            for j in range(self.size-1):
                if self.board[i][j] == self.board[i][j+1] or self.board[j][i] == self.board[j+1][i]:
                    return False
        return True
    
    def move_is_valid(self, direction):
        temp_game = Game2048()
        temp_game.board = np.copy(self.board)
        temp_game.move(direction)
        is_valid = not np.array_equal(temp_game.board, self.board)

        return is_valid
    
    def board_from_move(self, direction):
        temp_game = Game2048()
        temp_game.board = np.copy(self.get_board())
        new_tile = temp_game.fake_move(direction)

        return temp_game.get_board(), new_tile
    

    def get_board(self):
        return self.board