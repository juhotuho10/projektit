from game2048 import Game2048
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Replay memory to store state and reward pairs
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum size of the buffer
        self.buffer = []  # List to store the state-reward pairs

    def is_full(self):
        # Check if the buffer has reached its capacity
        return len(self.buffer) == self.capacity
    
    def push(self, state, reward):
        # Append a new state-reward pair to the buffer
        experience = (state, reward)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            # Remove the oldest entry if buffer is full
            self.buffer.pop(0)
    
    def sample(self):
        # Retrieve and reset the buffer
        state, reward = zip(*self.buffer)
        self.buffer = []
        return state, reward

# Neural network model with dual inputs (CNN for grid and Dense for features)
class DualInputModel(nn.Module):
    def __init__(self):
        super(DualInputModel, self).__init__()
        # Convolutional layers for grid processing
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        # Dense layer for additional features
        self.fc_features = nn.Linear(in_features=5, out_features=64)
        # Combined layers for final processing
        self.fc1 = nn.Linear(in_features=64 * 3 * 3 + 64, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=1)

    def forward(self, grid_input, features_input):
        # Process grid input through convolutional layers
        x1 = F.relu(self.conv1(grid_input))
        x1 = self.flatten(x1)
        # Process features input through dense layer
        x2 = F.relu(self.fc_features(features_input))
        # Combine and process through additional layers
        combined = torch.cat((x1, x2), dim=1)
        combined = F.relu(self.fc1(combined))
        output = self.output(combined)
        return output
    
# Class to manage the neural network model and replay memory
class torch_model():
    def __init__(self, memory_size = 1000):
        self.model = DualInputModel()  # Initialize the model
        self.memory = ReplayBuffer(memory_size)  # Initialize the memory buffer

    def data_from_memory(self):
        # Retrieve and prepare data from memory for training
        states, rewards = self.memory.sample()
        board_data, other_data = zip(*states)
        board_data = np.array(board_data, dtype=np.float32)
        other_data = np.array(other_data, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        print(f"max tile reached: {2 ** np.max(board_data)}")
        
        board_data = torch.from_numpy(board_data)
        other_data = torch.from_numpy(other_data)
        rewards = torch.from_numpy(rewards)
        return board_data, other_data, rewards

    def train_model(self, epochs, batch_size):

        board_data, other_data, rewards = self.data_from_memory()

        # Set the model to training mode
        self.model.train()

        # Define the optimizer and the loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):

            for i in range(0, len(board_data), batch_size):
                # Get the mini-batch
                batch_grid = board_data[i:i+batch_size]
                batch_features = other_data[i:i+batch_size]
                batch_labels = rewards[i:i+batch_size]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_grid, batch_features)
                loss = criterion(outputs.squeeze(), batch_labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

    def predict(self, grid_input, features_input):
        # predict a score from a given state
        self.model.eval()
        grid_input = np.array(grid_input, dtype=np.float32)
        features_input = np.array(features_input, dtype=np.float32)

        with torch.no_grad():  # Disable gradient computation
            grid_input = torch.from_numpy(grid_input)
            features_input = torch.from_numpy(features_input)
            
            predicted_value = self.model(grid_input, features_input)
            
        return predicted_value
        
    def save_model(self):
        # Save the trained model
        model_path = "2048_model.pt"
        torch.save(self.model, model_path)

    def load_model(self):
        # Load a pre-trained model
        model_path = "2048_model.pt"
        if os.path.exists(model_path):
            self.model = torch.load(model_path)
        else:
            print("Model path doesn't exists")

# Function to preprocess game board data
def get_data(curr_board: np.ndarray):

    curr_board = curr_board.astype(int)
    # taking a log2 of the board, so 0 = 0, 2 = 1, 4 = 2, 8 = 3 etc...
    log_board = np.where(curr_board > 0.9, np.log2(curr_board, where=curr_board > 0.9), 0).astype(np.int32)

    # Calculate various features from the board
    empty_count = sum(curr_board.flatten() == 0) / 16

    max_num_count = np.sum(curr_board == np.max(curr_board))

    mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
    max_mergeable = max([mergeable_rows, mergeable_cols])

    valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)
    valid_move_count = len(valid_moves)

    max_value_score = get_max_value_score(curr_board)

    log_board = np.array([log_board])
    other_data = np.array([empty_count, max_num_count, max_mergeable, valid_move_count, max_value_score], dtype=np.float32)
    return log_board, other_data

# recursive function to check the count of connecting chain of tiles 
def find_longest_chain(board, tile_value, i, j, visited):
    if (i, j) in visited or board[i][j] != tile_value:
        return 0

    visited.add((i, j))
    max_chain = 1  # Current tile is part of the chain

    # Directions: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for dx, dy in directions:
        x, y = i + dx, j + dy
        if 0 <= x < len(board) and 0 <= y < len(board) and board[x][y] == tile_value / 2:
            max_chain = max(max_chain, 1 + find_longest_chain(board, tile_value / 2, x, y, visited))

    return max_chain

# returns the maximum length of tiles that in the chain to be merged starting from the biggest tile
# 64 - 32 - 16 would return a count of 3
def calculate_longest_chain(board):
    max_chain_length = 0
    visited = set()

    max_value_positions = find_max_positions(board)

    for pos in max_value_positions:
        i, j = pos
        chain_length = find_longest_chain(board, board[i][j], i, j, visited)
        max_chain_length = max(max_chain_length, chain_length)

    return max_chain_length

# returns positions of max values
def find_max_positions(grid):
    max_val = np.max(grid)
    positions = np.where(grid == max_val)
    return list(zip(positions[0], positions[1]))

# Function to count the number of tiles that can be merged on the board
def count_mergeable_tiles(curr_board: np.ndarray) -> int:
    # Initialize counts for rows and columns
    mergeable_rows = 0
    mergeable_cols = 0

    # Helper function to count mergeable tiles in a single line
    def count_mergeable_in_line(line):
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

# Function to determine valid moves based on the current board state
def get_valid_moves(curr_board: np.ndarray, mergeable_rows: int, mergeable_cols: int):
    valid_directions = []

    # Add directions where merges are possible
    if mergeable_rows != 0:
        valid_directions.extend(["left", "right"])
    if mergeable_cols != 0:
        valid_directions.extend(["up", "down"])

    if len(set(valid_directions)) == 4:
        return ["up", "left", "down", "right"]

    sub_array = curr_board[1:3, 1:3]
    if np.any(sub_array == 0):
        return ["up", "left", "down", "right"]
    
    # Handle edge cases and return the set of valid moves
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

# Function to calculate a score based on the position of the maximum value tile
def get_max_value_score(curr_board: np.ndarray):
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

# Function to prepare data from different board states after possible moves
def data_from_boards(boards_after_moves: np.ndarray):
    board_data_list = []
    feature_data_list = []

    for board in boards_after_moves:
        log_board, other_data = get_data(board)

        board_data_list.append(log_board)
        feature_data_list.append(other_data)

    board_data_list = np.array(board_data_list, dtype=np.float32)
    feature_data_list = np.array(feature_data_list, dtype=np.float32)

    return board_data_list, feature_data_list

# Function to calculate the reward based on the board state
def board_reward(board: np.ndarray):

    # Calculate reward components
    chain_reward = calculate_longest_chain(board)

    max_value_position_score = get_max_value_score(board)

    max_num_reward = np.max(board)

    empty_spots = np.sum(board == 0)
    mergeable_rows, mergeable_cols = count_mergeable_tiles(board)

    max_merges = np.max([mergeable_rows, mergeable_cols])

    # Combine rewards into a total score
    total_reward = (max_num_reward * 1.2 + empty_spots * 2 + max_merges + chain_reward + max_value_position_score * 0.5) / 10

    return total_reward

# Function to find the best reward from all possible moves
def best_board_reward(game, curr_board):
    mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
    valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)

    if not valid_moves:
        return -10
    
    boards_after_moves = [game.board_from_move(move) for move in valid_moves]

    board_rewards = [board_reward(board) for board in boards_after_moves]

    best_reward = max(board_rewards)

    return best_reward

# Function to generate rewards for a series of moves
def generate_rewards(game, moves_taken):

    boards = [move[0] for move in moves_taken]

    board_rewards = [best_board_reward(game, board) for board in boards]

    return board_rewards

# Main function to gather data for training 
def gather_data():

    pred_model = torch_model(1000)

    for _ in range(50):
        while not pred_model.memory.is_full():
            game = Game2048()
            moves_taken = []
            boards_taken = []
            while True:
                curr_board = game.get_board()

                # all valis moves
                mergeable_rows, mergeable_cols = count_mergeable_tiles(curr_board)
                valid_moves = get_valid_moves(curr_board, mergeable_rows, mergeable_cols)

                if not valid_moves: # no possible moves = game ends
                    break
                
                # the boards after moves
                boards_after_moves = [game.board_from_move(move) for move in valid_moves]
                
                board_data_list, feature_data_list = data_from_boards(boards_after_moves)
                
                # score the boards with torch model
                predicted_scores = pred_model.predict(board_data_list, feature_data_list)

                max_index = np.argmax(predicted_scores)

                # take board with best score
                chosen_board = boards_after_moves[max_index]

                game.board = np.copy(chosen_board)

                # record data for training
                boards_taken.append(chosen_board)

                taken_board_data = board_data_list[max_index]
                taken_feature_data = feature_data_list[max_index]

                moves_taken.append([taken_board_data, taken_feature_data])

            # generate the actual rewards for boards that were taken
            rewards = generate_rewards(game, moves_taken)

            assert len(rewards) == len(moves_taken)

            for move, reward in zip(moves_taken, rewards):
                pred_model.memory.push(move, reward)

        # train the model with the boards taken and actual rewards
        pred_model.train_model(epochs=100, batch_size=64)

    pred_model.save_model()

if __name__ == "__main__":
    gather_data()