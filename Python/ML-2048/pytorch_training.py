from game2048 import Game2048
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from time import perf_counter as pc

# makes a replay memory with all the moves that have happened
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def is_full(self):
        return len(self.buffer) == self.capacity
    
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
        self.buffer = []
        return state, reward

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
        output = self.output(combined)

        return output

def get_data(curr_board: np.ndarray):

    curr_board = curr_board.astype(int)
    # taking a log2 of the board, so 0 = 0, 2 = 1, 4 = 2, 8 = 3 etc...

    log_board = np.where(curr_board > 0.9, np.log2(curr_board, where=curr_board > 0.9), 0).astype(np.int32)

    if np.max(log_board) > 15:
        print(curr_board)
        print(log_board)
        assert False

    empty_count = sum(curr_board.flatten() == 0) / 16

    max_num_count = np.sum(curr_board == np.max(curr_board))

    mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
    max_mergeable = max([mergeable_rows, mergeable_cols])

    valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)
    valid_move_count = len(valid_moves)

    max_value_score = get_max_value_score(curr_board)

    # Ensure the grid_input has the correct shape
    #log_board = np.expand_dims(log_board, axis=0) 
    log_board = np.array([log_board])

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
    
def data_from_boards(boards_after_moves):
    board_data_list = []
    feature_data_list = []

    for board in boards_after_moves:
        log_board, other_data = get_data(board)

        board_data_list.append(log_board)
        feature_data_list.append(other_data)

    board_data_list = np.array(board_data_list, dtype=np.float32)
    feature_data_list = np.array(feature_data_list, dtype=np.float32)

    return board_data_list, feature_data_list

def board_reward(board):

    max_num_reward = np.max(board)

    empty_spots = np.sum(board == 0)

    empty_spot_reward = (empty_spots / 10) + 0.1

    total_reward = max_num_reward * empty_spot_reward

    return total_reward

def best_board_reward(game, curr_board):
    mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
    valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)

    if not valid_moves:
        return -10
    
    boards_after_moves = [game.board_from_move(move) for move in valid_moves]

    board_rewards = [board_reward(board) for board in boards_after_moves]

    best_reward = max(board_rewards)

    return best_reward


def generate_rewards(game, moves_taken):

    boards = [move[0] for move in moves_taken]

    board_rewards = [best_board_reward(game, board) for board in boards]

    return board_rewards
    
def gather_data(pred_model):
    memory = ReplayBuffer(1000)

    while True:
        start = pc()
        while not memory.is_full():
            game = Game2048()
            moves_taken = []
            boards_taken = []
            while True:
                curr_board = game.get_board()
                mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
                valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)

                if not valid_moves:
                    break

                boards_after_moves = [game.board_from_move(move) for move in valid_moves]
                
                board_data_list, feature_data_list = data_from_boards(boards_after_moves)

                predicted_scores = predict(pred_model, board_data_list, feature_data_list)

                max_index = np.argmax(predicted_scores)

                chosen_board = boards_after_moves[max_index]

                game.board = np.copy(chosen_board)

                assert np.array_equal(game.get_board(), chosen_board)

                boards_taken.append(chosen_board)

                taken_board_data = board_data_list[max_index]
                taken_feature_data = feature_data_list[max_index]

                moves_taken.append([taken_board_data, taken_feature_data])

            rewards = generate_rewards(game, moves_taken)

            assert len(rewards) == len(moves_taken)

            for move, reward in zip(moves_taken, rewards):
                memory.push(move, reward)

        print(pc() -  start)

        train_model(pred_model, memory, epochs=100, batch_size=64)


def train_model(model, memory, epochs, batch_size):

    states, rewards = memory.sample()

    board_data, other_data = zip(*states)

    board_data = np.array(board_data, dtype=np.float32)
    other_data = np.array(other_data, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)

    print(np.max(board_data))

    # Set the model to training mode
    model.train()

    # Convert to PyTorch tensors
    board_data = torch.from_numpy(board_data)
    other_data = torch.from_numpy(other_data)
    rewards = torch.from_numpy(rewards)


    # Define the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(board_data), batch_size):
            # Get the mini-batch
            batch_grid = board_data[i:i+batch_size]
            batch_features = other_data[i:i+batch_size]
            batch_labels = rewards[i:i+batch_size]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_grid, batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #average_loss = total_loss / (len(board_data) // batch_size)
        #print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')


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
        label = 1  
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
