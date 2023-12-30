import numpy as np
import random

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty_tiles = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty_tiles:
            i, j = random.choice(empty_tiles)
            self.board[i][j] = random.choice([2, 4])

    def compress(self, grid):

        for i in range(4):
            line = grid[i]
            line = line[line != 0]

            pad_len = 4 - len(line)

            line = np.pad(line, (0, pad_len), 'constant')

            grid[i] = line

        return grid

    def merge(self, grid):
        for i in range(4):
            for j in range(3):
                if grid[i][j] == grid[i][j+1]:
                    grid[i][j] *= 2
                    grid[i][j+1] = 0
        return grid

    def reverse(self, grid):
        return np.flip(grid, axis=1)

    def transpose(self, grid):
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


    def move_board(self, direction):
        assert direction in ["left", "right", "up", "down"]

        self.move(direction)

        self.add_new_tile()
          
    def is_game_over(self):
        if any(0 in row for row in self.board):
            return False
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
        temp_game.board = np.copy(self.board)
        temp_game.move_board(direction)

        return temp_game.get_board()
    

    def get_board(self):
        return self.board