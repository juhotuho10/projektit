from game2048 import Game2048
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# makes a replay memory with all the moves that have happened
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    #appends to the memory
    def push(self, state, reward):
        experience = (state, reward)
        self.buffer.append(experience)
        # if memory is "full" we remove from the begining
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    # grabs the memory
    def sample(self):
        state, reward = zip(*self.buffer)
        return np.stack(state), np.array(reward)

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
        output = F.sigmoid(self.output(combined))

        return output


def get_data(curr_board: np.ndarray):

    # taking a log2 of the board, so 0 = 0, 2 = 1, 4 = 2, 8 = 3 etc...
    log_board = np.where(curr_board != 0, np.log2(curr_board), 0).astype(np.float32)
    
    empty_count = sum(curr_board.flatten() == 0) / 16

    max_num_count = np.sum(curr_board == np.max(curr_board))

    mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
    max_mergeable = max([mergeable_rows, mergeable_cols])

    valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)
    valid_move_count = len(valid_moves)

    max_value_score = get_max_value_score(curr_board)

    # Ensure the grid_input has the correct shape
    log_board = np.expand_dims(log_board, axis=0) 

    other_data = np.array([empty_count, max_num_count, max_mergeable, valid_move_count, max_value_score], dtype=np.float32)
    return log_board, other_data

def count_mergeable_tiles(curr_board):
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
    for row in curr_board:
        mergeable_rows += count_mergeable_in_line(row)

    # Count mergeable tiles in columns
    for col in curr_board.T:  # Transpose to iterate over columns
        mergeable_cols += count_mergeable_in_line(col)

    return mergeable_rows, mergeable_cols

def get_valid_moves(curr_board, mergeable_rows, mergeable_cols):
    valid_directions = []

    if mergeable_rows != 0:
        valid_directions.extend(["left", "right"])
    if mergeable_cols != 0:
        valid_directions.extend(["up", "down"])

    if len(set(valid_directions)) == 4:
        return ["up", "left", "down", "right"]

    sub_array = curr_board[1:3, 1:3]
    if np.any(sub_array == 0):
        return ["up", "left", "down", "right"]
    
    # if all the tiles in a row / col are 0, it's not a valid move since it doesnt change the game state at all
    columns_to_keep = ~np.all(curr_board == 0, axis=0)
    modified_horisontal = curr_board[:, columns_to_keep]

    if np.any(modified_horisontal[0] == 0):
        valid_directions.append("up")
    if np.any(modified_horisontal[-1] == 0):
        valid_directions.append("down")

    rows_to_keep = ~np.all(curr_board == 0, axis=1)
    modified_vertical = curr_board[rows_to_keep]

    if np.any(modified_vertical[:, 0] == 0):
        valid_directions.append("left")
    if np.any(modified_vertical[:, -1] == 0):
        valid_directions.append("right")

    return list(set(valid_directions))

def get_max_value_score(curr_board):
    max_value = np.max(curr_board)

    # checks if max value is in a corner
    # [(x coords), (y coords)]
    corner_coordinates = [(0,0,3,3), (0,3,0,3)]
    corner_values = curr_board[tuple(zip(corner_coordinates))]
    if np.any(corner_values == max_value):
        return 2
    
    # check if max value is in the side
    elif np.any(curr_board[[0,3]] == max_value) or np.any(curr_board[:,[0,3]] == max_value):
        return 1
    # otherwise max value must be in the middle 
    else:
        return 0

def predict(pred_model, grid_input, features_input):

    grid_input = np.array(grid_input, dtype=np.float32)
    features_input = np.array(features_input, dtype=np.float32)

    with torch.no_grad():  # Disable gradient computation
        grid_input = torch.from_numpy(grid_input)
        features_input = torch.from_numpy(features_input)
        
        predicted_value = pred_model(grid_input, features_input)

        return predicted_value
    
def gather_data(pred_model):
    for _ in range(100):
        game = Game2048()
        done = False
        while True:
            curr_board = game.get_board()
            mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
            valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)

            if not valid_moves:
                print("game done")
                break

            boards_after_moves = []
            for move in valid_moves:
                nwe_board = game.board_from_move(move)
                boards_after_moves.append(nwe_board)

            board_data_list = []
            feature_data_list = []

            for board in boards_after_moves:
                log_board, other_data = get_data(board)
                board_data_list.append(log_board)
                feature_data_list.append(other_data)

            board_data_list = np.array(board_data_list, dtype=np.float32)
            feature_data_list = np.array(feature_data_list, dtype=np.float32)

            predicted_scores = predict(pred_model, board_data_list, feature_data_list)

            max_index = np.argmax(predicted_scores)

            best_move = valid_moves[max_index]

            game.move_board(best_move)
                    

# Example model predictions for each sample
model = DualInputModel()
model.eval()  # Set the model to evaluation mode

gather_data(model)

def generate_dataset(num_samples=1000):
    x_grid_list = []
    x_features_list = []
    y_list = []

    for _ in range(num_samples):
        grid_data, feature_data = get_data()
        label = 1  # Random float between 0 and 1
        x_grid_list.append(grid_data)
        x_features_list.append(feature_data)
        y_list.append(label)

    # Convert lists to NumPy arrays
    x_grid = np.array(x_grid_list, dtype=np.float32)
    x_features = np.array(x_features_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return x_grid, x_features, y

'''
# Generate the dataset
x_train_grid, x_train_features, y_train = generate_dataset(1000)

# Convert to PyTorch tensors
x_train_grid = torch.from_numpy(x_train_grid)
x_train_features = torch.from_numpy(x_train_features)
y_train = torch.from_numpy(y_train)

# Create the model
model = DualInputModel()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(x_train_grid), batch_size):
        # Get the mini-batch
        batch_grid = x_train_grid[i:i+batch_size]
        batch_features = x_train_features[i:i+batch_size]
        batch_labels = y_train[i:i+batch_size]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_grid, batch_features)
        loss = criterion(outputs.squeeze(), batch_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')'''
