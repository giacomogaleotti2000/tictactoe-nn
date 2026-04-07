# tictactoe-nn

A small neural-network Tic-Tac-Toe project built with PyTorch.

This repository trains an agent to play Tic-Tac-Toe using a Deep Q-Learning style setup with:

- a Tic-Tac-Toe environment
- a feedforward neural network
- experience replay
- a target network
- epsilon-greedy exploration
- evaluation against a random opponent

## Project Structure

### `TicTacToe.py`

This is the core file of the project.

It contains:

- `TicTacToeEnv`: the game environment
- `QNetwork`: the neural network that predicts Q-values for the 9 board positions
- `ReplayBuffer`: experience replay memory
- `select_action()`: epsilon-greedy move selection
- `train_step()`: one optimization step
- `train()`: the main training loop
- `evaluate_vs_random()`: evaluation against a random opponent

### `ResumeTraining.py`

This file contains `train_resume()`, which continues training from an already trained model, target network, optimizer, and replay buffer.

It is useful for extending training after the initial run.

### `main.py`

This is the main entry point.

It:

1. runs the initial training phase
2. resumes training for additional episodes
3. saves the final model weights to `model_name.pth`

## How It Works

The board is represented as a vector of 9 cells:

- `0` = empty
- `1` = agent move
- `-1` = opponent move

The neural network takes the board as input and outputs 9 Q-values, one for each possible action.

During training:

- the agent chooses actions with epsilon-greedy exploration
- the opponent plays randomly
- transitions are stored in the replay buffer
- the model is updated using batches sampled from memory
- a target network is updated periodically for more stable learning

## Training

To run training:

```bash
python main.py
