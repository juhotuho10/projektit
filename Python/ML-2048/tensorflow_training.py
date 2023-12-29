from game2048 import Game2048

import numpy as np

game = Game2048()

print(game.get_board())
game.move_board("left")
print(game.get_board())

def get_data():
    board = np.array(game.get_board())
    

    empty_count = sum(board.flatten() == 0)
    max_num_count = np.sum(board == np.max(board))

    mergeable_rows, mergeable_cols = count_mergeable_tiles()
    max_mergeable = max([mergeable_rows, mergeable_cols])
    valid_moves = get_valid_moves(mergeable_rows, mergeable_cols)
    valid_move_count = len(valid_moves)

    max_value_score = get_max_value_score()

    other_data = np.array([empty_count, max_num_count, max_mergeable, valid_move_count, max_value_score])

    return board, 




def count_mergeable_tiles():
    board = game.get_board()
    mergeable_rows = 0
    mergeable_cols = 0

    # Function to count mergeable tiles in a line using NumPy
    def count_mergeable_in_line(line):
        # Remove zeros and find differences between adjacent elements
        non_zero_line = line[line != 0]
        diffs = np.diff(non_zero_line)
        # Count where adjacent elements are the same (diff is 0)
        return np.sum(diffs == 0)

    # Count mergeable tiles in rows
    for row in board:
        mergeable_rows += count_mergeable_in_line(row)

    # Count mergeable tiles in columns
    for col in board.T:  # Transpose to iterate over columns
        mergeable_cols += count_mergeable_in_line(col)

    return mergeable_rows, mergeable_cols

def get_valid_moves(mergeable_rows, mergeable_cols):
    board = np.array(game.get_board())
    valid_directions = []

    if mergeable_rows != 0 and mergeable_cols !=0:
        return ["up", "left", "down", "right"]

    if mergeable_rows != 0:
        valid_directions.extend(["left", "right"])
    if mergeable_cols != 0:
        valid_directions.extend(["up", "down"])

    sub_array = board[1:3, 1:3]
    if np.any(sub_array == 0):
        return ["up", "left", "down", "right"]
    

    # if all the tiles in a row / col are 0, it's not a valid move since it doesnt change the game state at all
    columns_to_keep = ~np.all(board == 0, axis=0)
    modified_horisontal = board[:, columns_to_keep]

    if np.any(modified_horisontal[0] == 0):
        valid_directions.append("up")
    if np.any(modified_horisontal[-1] == 0):
        valid_directions.append("down")

    rows_to_keep = ~np.all(board == 0, axis=1)
    modified_vertical = board[rows_to_keep]

    if np.any(modified_vertical[:, 0] == 0):
        valid_directions.append("left")
    if np.any(modified_vertical[:, -1] == 0):
        valid_directions.append("right")

    return list(set(valid_directions))

def get_max_value_score():
    board = np.array(game.get_board())
    max_value = np.max(board)

    # checks if max value is in a corner
    corner_coordinates = [(0,0,3,3), (0,3,0,3)]
    corner_values = board[tuple(zip(corner_coordinates))]
    if np.any(corner_values == max_value):
        return 2
    
    # check if max value is in the side
    elif np.any(board[[0,3]] == max_value) or np.any(board[:,[0,3]] == max_value):
        return 1
    # otherwise max value must be in the middle 
    else:
        return 0


get_data()
print(count_mergeable_tiles())