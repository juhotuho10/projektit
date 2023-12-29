from game2048 import Game2048

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

game = Game2048()

print(game.get_board())
game.move_board("left")
print(game.get_board())

class DualInputModel(nn.Module):
    def __init__(self):
        super(DualInputModel, self).__init__()

        # CNN branch for the game grid
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 2))
        self.flatten = nn.Flatten()

        # Dense branch for the features
        self.fc_features = nn.Linear(in_features=5, out_features=64)

        # The size of the flattened CNN output needs to be calculated correctly
        # Assuming the input size is (1, 4, 4), after Conv2d(2x2) and flatten, it becomes 64 * (3 * 3)
        cnn_output_size = 64 * 3 * 3  # Adjust this based on your actual CNN output

        # Combined layers
        self.fc1 = nn.Linear(in_features=cnn_output_size + 64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, grid_input, features_input):
        # CNN branch
        x1 = F.relu(self.conv1(grid_input))
        x1 = self.flatten(x1)

        # Dense branch
        x2 = F.relu(self.fc_features(features_input))

        # Combine branches
        combined = torch.cat((x1, x2), dim=1)

        # Additional dense layers
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        output = torch.sigmoid(self.output(combined))

        return output


def get_data():
    board = np.array(game.get_board())
    
    # taking a log2 of the board, so 0 = 0, 2 = 1, 4 = 2, 8 = 3 etc...
    log_board = np.log2(board, where=(board!=0)).astype(int).astype(np.float32)
    
    empty_count = sum(board.flatten() == 0)

    max_num_count = np.sum(board == np.max(board))

    mergeable_rows, mergeable_cols = count_mergeable_tiles()
    max_mergeable = max([mergeable_rows, mergeable_cols])

    valid_moves = get_valid_moves(mergeable_rows, mergeable_cols)
    valid_move_count = len(valid_moves)

    max_value_score = get_max_value_score()

    # Ensure the grid_input has the correct shape
    log_board = np.expand_dims(log_board, axis=0) 
    log_board = np.expand_dims(log_board, axis=0) 


    other_data = np.array([empty_count, max_num_count, max_mergeable, valid_move_count, max_value_score], dtype=np.float32)
    other_data = np.expand_dims(other_data, axis=0) 

    return log_board, other_data

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

    if mergeable_rows != 0:
        valid_directions.extend(["left", "right"])
    if mergeable_cols != 0:
        valid_directions.extend(["up", "down"])

    if len(set(valid_directions)) == 4:
        return ["up", "left", "down", "right"]

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
    # [(x coords), (y coords)]
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

# Example model predictions for each sample
model = DualInputModel()
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Disable gradient computation
    grid_input, features_input = get_data()

    grid_input = torch.from_numpy(grid_input)
    features_input = torch.from_numpy(features_input)
    
    predicted_value = model(grid_input, features_input)
    print(f"Predicted Value: {predicted_value.item()}")