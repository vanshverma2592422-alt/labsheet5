import random
import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# COMMON SETUP AND HELPER FUNCTIONS
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_frozenlake():
    """Create a FrozenLake environment."""
    return gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )


def train_q_learning(
    env,
    episodes=3000,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
):
    """Train a tabular Q-Learning agent."""
    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            done = terminated or truncated

            best_next_value = (
                0 if done else np.max(q_table[next_state])
            )

            q_table[state, action] += learning_rate * (
                reward
                + gamma * best_next_value
                - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_rewards


def evaluate_q_agent(env, q_table, episodes=100):
    """Evaluate a trained Q-Learning agent."""
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    return float(np.mean(rewards))


# ============================================================
# PROGRAM 1: Install and Configure Gymnasium
# ============================================================

# Install Gymnasium using:
# pip install gymnasium

import gymnasium as gym

print("Gymnasium version:", gym.__version__)


# ============================================================
# PROGRAM 2: Create and Execute a Simple RL Environment
# ============================================================

env = create_frozenlake()

state, info = env.reset(seed=42)
action = env.action_space.sample()

next_state, reward, terminated, truncated, info = (
    env.step(action)
)

env.close()


# ============================================================
# PROGRAM 3: Explore Observation Space and Action Space
# ============================================================

env = create_frozenlake()

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Number of States:", env.observation_space.n)
print("Number of Actions:", env.action_space.n)

env.close()


# ============================================================
# PROGRAM 4: Display States, Actions, Rewards and Termination
# ============================================================

env = create_frozenlake()

state, info = env.reset(seed=42)

for step_number in range(10):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(
        "Step:", step_number,
        "State:", state,
        "Action:", action,
        "Reward:", reward,
        "Terminated:", terminated,
        "Truncated:", truncated
    )

    state = next_state

    if terminated or truncated:
        break

env.close()


# ============================================================
# PROGRAM 5: Simulate Random Actions in FrozenLake
# ============================================================

env = create_frozenlake()

state, info = env.reset(seed=42)
total_reward = 0

for _ in range(100):
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    if terminated or truncated:
        break

print("Total Random-Agent Reward:", total_reward)

env.close()


# ============================================================
# PROGRAM 6: Implement Q-Learning for FrozenLake
# ============================================================

env = create_frozenlake()

q_table, training_rewards = train_q_learning(
    env,
    episodes=3000,
    learning_rate=0.1,
    gamma=0.99
)

env.close()


# ============================================================
# PROGRAM 7: Initialize and Update the Q-Table
# ============================================================

env = create_frozenlake()

q_table_example = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

state, info = env.reset(seed=42)
action = env.action_space.sample()

next_state, reward, terminated, truncated, info = (
    env.step(action)
)

done = terminated or truncated
alpha = 0.1
gamma = 0.99

next_value = (
    0 if done else np.max(q_table_example[next_state])
)

q_table_example[state, action] += alpha * (
    reward
    + gamma * next_value
    - q_table_example[state, action]
)

env.close()


# ============================================================
# PROGRAM 8: Train an Agent for Multiple Episodes
# ============================================================

env = create_frozenlake()

q_table, episode_rewards = train_q_learning(
    env,
    episodes=5000,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=1.0
)

env.close()


# ============================================================
# PROGRAM 9: Display the Learned Q-Table
# ============================================================

q_table_df = pd.DataFrame(
    q_table,
    columns=["Left", "Down", "Right", "Up"]
)

print(q_table_df)


# ============================================================
# PROGRAM 10: Evaluate the Trained Q-Learning Agent
# ============================================================

env = create_frozenlake()

average_reward = evaluate_q_agent(
    env,
    q_table,
    episodes=100
)

print("Average Evaluation Reward:", average_reward)

env.close()


# ============================================================
# PROGRAM 11: Plot Cumulative Rewards During Training
# ============================================================

cumulative_rewards = np.cumsum(training_rewards)

plt.figure(figsize=(8, 5))
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards During Q-Learning")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 12: Study Different Learning Rates
# ============================================================

learning_rates = [0.01, 0.1, 0.5]
learning_rate_results = {}

for alpha_value in learning_rates:
    env = create_frozenlake()

    q_values, rewards = train_q_learning(
        env,
        episodes=3000,
        learning_rate=alpha_value,
        gamma=0.99
    )

    learning_rate_results[alpha_value] = rewards
    env.close()

plt.figure(figsize=(8, 5))

for alpha_value, rewards in learning_rate_results.items():
    plt.plot(
        pd.Series(rewards).rolling(100).mean(),
        label=f"Learning rate = {alpha_value}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate")
plt.legend()
plt.show()


# ============================================================
# PROGRAM 13: Study Different Discount Factor Values
# ============================================================

gamma_values = [0.5, 0.9, 0.99]
gamma_results = {}

for gamma_value in gamma_values:
    env = create_frozenlake()

    q_values, rewards = train_q_learning(
        env,
        episodes=3000,
        learning_rate=0.1,
        gamma=gamma_value
    )

    gamma_results[gamma_value] = rewards
    env.close()

plt.figure(figsize=(8, 5))

for gamma_value, rewards in gamma_results.items():
    plt.plot(
        pd.Series(rewards).rolling(100).mean(),
        label=f"Gamma = {gamma_value}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Effect of Discount Factor")
plt.legend()
plt.show()


# ============================================================
# PROGRAM 14: Compare Exploration and Exploitation
# ============================================================

epsilon_values = [0.1, 0.5, 1.0]
epsilon_results = {}

for epsilon_value in epsilon_values:
    env = create_frozenlake()

    q_values, rewards = train_q_learning(
        env,
        episodes=3000,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=epsilon_value,
        epsilon_decay=0.999
    )

    epsilon_results[epsilon_value] = rewards
    env.close()

plt.figure(figsize=(8, 5))

for epsilon_value, rewards in epsilon_results.items():
    plt.plot(
        pd.Series(rewards).rolling(100).mean(),
        label=f"Initial epsilon = {epsilon_value}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Exploration and Exploitation")
plt.legend()
plt.show()


# ============================================================
# PROGRAM 15: Implement an Epsilon-Greedy Policy
# ============================================================

def epsilon_greedy_action(q_table, state, epsilon, action_space):
    """Choose a random or greedy action."""
    if random.random() < epsilon:
        return action_space.sample()

    return int(np.argmax(q_table[state]))


env = create_frozenlake()
state, info = env.reset(seed=42)

selected_action = epsilon_greedy_action(
    q_table,
    state,
    epsilon=0.1,
    action_space=env.action_space
)

env.close()


# ============================================================
# PROGRAM 16: Design a Simple Grid World Environment
# ============================================================

class GridWorld:
    """A small grid environment with start and goal positions."""

    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start

        # Actions: up, down, left, right
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        row_change, col_change = self.actions[action]

        row = np.clip(row + row_change, 0, self.size - 1)
        col = np.clip(col + col_change, 0, self.size - 1)

        self.state = (int(row), int(col))
        done = self.state == self.goal
        reward = 1.0 if done else -0.01

        return self.state, reward, done


grid = GridWorld()


# ============================================================
# PROGRAM 17: Train Q-Learning in the Custom Grid World
# ============================================================

grid = GridWorld(size=4)

grid_q_table = np.zeros((16, 4))
grid_rewards = []

alpha = 0.1
gamma = 0.99
epsilon = 1.0

for episode in range(3000):
    state_tuple = grid.reset()
    state = state_tuple[0] * grid.size + state_tuple[1]

    total_reward = 0

    for _ in range(100):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(grid_q_table[state]))

        next_tuple, reward, done = grid.step(action)

        next_state = (
            next_tuple[0] * grid.size + next_tuple[1]
        )

        next_value = (
            0 if done else np.max(grid_q_table[next_state])
        )

        grid_q_table[state, action] += alpha * (
            reward
            + gamma * next_value
            - grid_q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    grid_rewards.append(total_reward)
    epsilon = max(0.01, epsilon * 0.995)


# ============================================================
# PROGRAM 18: Visualize the Optimal Grid World Path
# ============================================================

grid = GridWorld(size=4)
state_tuple = grid.reset()

path = [state_tuple]

for _ in range(20):
    state = (
        state_tuple[0] * grid.size + state_tuple[1]
    )

    action = int(np.argmax(grid_q_table[state]))

    state_tuple, reward, done = grid.step(action)
    path.append(state_tuple)

    if done:
        break

grid_visual = np.zeros((grid.size, grid.size))

for row, col in path:
    grid_visual[row, col] = 1

plt.figure(figsize=(6, 6))
plt.imshow(grid_visual, cmap="Blues")
plt.xticks(range(grid.size))
plt.yticks(range(grid.size))
plt.title("Optimal Grid World Path")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 19: Compare FrozenLake and Grid World
# ============================================================

env = create_frozenlake()

frozenlake_score = evaluate_q_agent(
    env,
    q_table,
    episodes=100
)

env.close()

grid = GridWorld(size=4)
grid_test_rewards = []

for _ in range(100):
    state_tuple = grid.reset()
    total_reward = 0

    for _ in range(50):
        state = (
            state_tuple[0] * grid.size + state_tuple[1]
        )

        action = int(np.argmax(grid_q_table[state]))
        state_tuple, reward, done = grid.step(action)

        total_reward += reward

        if done:
            break

    grid_test_rewards.append(total_reward)

print("FrozenLake Average Reward:", frozenlake_score)
print("Grid World Average Reward:", np.mean(grid_test_rewards))


# ============================================================
# PROGRAM 20: Analyze Q-Learning Convergence
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    pd.Series(episode_rewards).rolling(100).mean()
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Convergence")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 21: Prepare Libraries for DQN
# ============================================================

# Install required packages:
# pip install torch gymnasium numpy matplotlib pandas
#
# DQN uses PyTorch for neural networks and Gymnasium
# for the CartPole environment.

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# PROGRAM 22: Implement a Basic Deep Q-Network
# ============================================================

class DQN(nn.Module):
    """Neural network that estimates Q-values."""

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, states):
        return self.network(states)


class ReplayBuffer:
    """Store and sample past transitions."""

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


def train_dqn(
    episodes=150,
    replay_capacity=10000,
    use_target_network=True
):
    """Train DQN on CartPole and return the trained model."""

    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size).to(DEVICE)
    target_net = DQN(state_size, action_size).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(
        policy_net.parameters(),
        lr=0.001
    )

    loss_function = nn.MSELoss()
    replay_buffer = ReplayBuffer(replay_capacity)

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    episode_rewards = []
    episode_times = []

    for episode in range(episodes):
        start_time = time.time()

        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                with torch.no_grad():
                    action = int(
                        policy_net(state_tensor).argmax(dim=1).item()
                    )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            replay_buffer.add((
                state,
                action,
                reward,
                next_state,
                done
            ))

            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                ) = replay_buffer.sample(batch_size)

                states_tensor = torch.tensor(
                    states, device=DEVICE
                )

                actions_tensor = torch.tensor(
                    actions, device=DEVICE
                ).unsqueeze(1)

                rewards_tensor = torch.tensor(
                    rewards, device=DEVICE
                )

                next_states_tensor = torch.tensor(
                    next_states, device=DEVICE
                )

                dones_tensor = torch.tensor(
                    dones, device=DEVICE
                )

                current_q = policy_net(
                    states_tensor
                ).gather(1, actions_tensor).squeeze(1)

                with torch.no_grad():
                    if use_target_network:
                        next_q = target_net(
                            next_states_tensor
                        ).max(dim=1)[0]
                    else:
                        next_q = policy_net(
                            next_states_tensor
                        ).max(dim=1)[0]

                    target_q = rewards_tensor + gamma * (
                        1 - dones_tensor
                    ) * next_q

                loss = loss_function(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    policy_net.parameters(), 10.0
                )

                optimizer.step()

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        episode_rewards.append(total_reward)
        episode_times.append(time.time() - start_time)

        # Update target network periodically.
        if use_target_network and (episode + 1) % 10 == 0:
            target_net.load_state_dict(
                policy_net.state_dict()
            )

    env.close()

    return policy_net, episode_rewards, episode_times


# ============================================================
# PROGRAM 23: Train DQN on CartPole
# ============================================================

dqn_model, dqn_rewards, dqn_times = train_dqn(
    episodes=150
)


# ============================================================
# PROGRAM 24: Plot Episode-Wise DQN Rewards
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(dqn_rewards, label="Episode Reward")

plt.plot(
    pd.Series(dqn_rewards).rolling(10).mean(),
    label="10-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Rewards")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 25: Evaluate the Trained DQN Agent
# ============================================================

def evaluate_dqn(model, episodes=10):
    """Evaluate a DQN model without exploration."""
    env = gym.make("CartPole-v1")
    rewards = []

    for _ in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                action = int(
                    model(state_tensor).argmax(dim=1).item()
                )

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))


dqn_evaluation_score = evaluate_dqn(
    dqn_model,
    episodes=10
)

print("DQN Evaluation Average Reward:", dqn_evaluation_score)


# ============================================================
# PROGRAM 26: Compare Q-Learning and DQN Performance
# ============================================================

env = create_frozenlake()

frozenlake_q_table, frozenlake_rewards = train_q_learning(
    env,
    episodes=3000
)

frozenlake_evaluation = evaluate_q_agent(
    env,
    frozenlake_q_table,
    episodes=100
)

env.close()

performance_comparison = pd.DataFrame({
    "Algorithm": ["Q-Learning on FrozenLake", "DQN on CartPole"],
    "Average Reward": [
        frozenlake_evaluation,
        dqn_evaluation_score
    ]
})

print(performance_comparison)

# Note: These scores come from different environments,
# so they are not a direct apples-to-apples comparison.


# ============================================================
# PROGRAM 27: Analyze the Effect of Replay Memory
# ============================================================

replay_memory_results = {}

for memory_size in [1000, 10000]:
    model, rewards, times = train_dqn(
        episodes=100,
        replay_capacity=memory_size
    )

    replay_memory_results[memory_size] = rewards

plt.figure(figsize=(8, 5))

for memory_size, rewards in replay_memory_results.items():
    plt.plot(
        pd.Series(rewards).rolling(10).mean(),
        label=f"Replay Capacity = {memory_size}"
    )

plt.xlabel("Episode")
plt.ylabel("Moving Average Reward")
plt.title("Effect of Replay Memory")
plt.legend()
plt.show()


# ============================================================
# PROGRAM 28: Study the Role of the Target Network
# ============================================================

target_network_results = {}

for target_enabled in [True, False]:
    model, rewards, times = train_dqn(
        episodes=100,
        use_target_network=target_enabled
    )

    label = (
        "With Target Network"
        if target_enabled
        else "Without Target Network"
    )

    target_network_results[label] = rewards

plt.figure(figsize=(8, 5))

for label, rewards in target_network_results.items():
    plt.plot(
        pd.Series(rewards).rolling(10).mean(),
        label=label
    )

plt.xlabel("Episode")
plt.ylabel("Moving Average Reward")
plt.title("Effect of Target Network")
plt.legend()
plt.show()


# ============================================================
# PROGRAM 29: Save the Trained DQN Model
# ============================================================

torch.save(
    dqn_model.state_dict(),
    "cartpole_dqn_model.pth"
)


# ============================================================
# PROGRAM 30: Load the Saved DQN Model and Test It
# ============================================================

test_env = gym.make("CartPole-v1")

loaded_model = DQN(
    test_env.observation_space.shape[0],
    test_env.action_space.n
).to(DEVICE)

loaded_model.load_state_dict(
    torch.load(
        "cartpole_dqn_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

loaded_model.eval()

loaded_model_score = evaluate_dqn(
    loaded_model,
    episodes=10
)

print("Loaded Model Average Reward:", loaded_model_score)

test_env.close()


# ============================================================
# PROGRAM 31: Compare Cumulative Rewards of RL Algorithms
# ============================================================

q_cumulative = np.cumsum(frozenlake_rewards)
dqn_cumulative = np.cumsum(dqn_rewards)

plt.figure(figsize=(8, 5))

plt.plot(
    q_cumulative,
    label="Q-Learning: FrozenLake"
)

plt.plot(
    dqn_cumulative,
    label="DQN: CartPole"
)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards of RL Algorithms")
plt.legend()
plt.show()

# These curves use different environments and reward scales.


# ============================================================
# PROGRAM 32: Visualize the Learning Curve
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    pd.Series(dqn_rewards).rolling(10).mean()
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN Learning Curve")
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 33: Compare Training Time and Convergence
# ============================================================

q_start_time = time.time()

env = create_frozenlake()

q_timing_table, q_timing_rewards = train_q_learning(
    env,
    episodes=3000
)

q_training_time = time.time() - q_start_time
env.close()

dqn_start_time = time.time()

dqn_timing_model, dqn_timing_rewards, dqn_episode_times = (
    train_dqn(episodes=100)
)

dqn_training_time = time.time() - dqn_start_time

training_time_comparison = pd.DataFrame({
    "Algorithm": ["Q-Learning", "DQN"],
    "Training Time (seconds)": [
        q_training_time,
        dqn_training_time
    ]
})

print(training_time_comparison)

# Training times depend on hardware and configuration.
# The environments differ, so convergence is not directly
# comparable using raw rewards alone.


# ============================================================
# PROGRAM 34: Analyze the Impact of Hyperparameters
# ============================================================

hyperparameter_results = []

for alpha_value in [0.05, 0.1, 0.5]:
    for gamma_value in [0.8, 0.95, 0.99]:
        env = create_frozenlake()

        q_values, rewards = train_q_learning(
            env,
            episodes=2000,
            learning_rate=alpha_value,
            gamma=gamma_value
        )

        evaluation_score = evaluate_q_agent(
            env,
            q_values,
            episodes=50
        )

        hyperparameter_results.append({
            "Learning Rate": alpha_value,
            "Gamma": gamma_value,
            "Average Evaluation Reward": evaluation_score
        })

        env.close()

hyperparameter_df = pd.DataFrame(
    hyperparameter_results
)

print(hyperparameter_df)

plt.figure(figsize=(8, 5))

for alpha_value in [0.05, 0.1, 0.5]:
    subset = hyperparameter_df[
        hyperparameter_df["Learning Rate"] == alpha_value
    ]

    plt.plot(
        subset["Gamma"],
        subset["Average Evaluation Reward"],
        marker="o",
        label=f"Learning Rate = {alpha_value}"
    )

plt.xlabel("Discount Factor (Gamma)")
plt.ylabel("Average Evaluation Reward")
plt.title("Impact of Q-Learning Hyperparameters")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# PROGRAM 35: Comparative Report of RL Algorithms
# ============================================================

final_report = pd.DataFrame({
    "Algorithm": [
        "Q-Learning",
        "Deep Q-Network"
    ],
    "Environment": [
        "FrozenLake",
        "CartPole"
    ],
    "Average Evaluation Reward": [
        frozenlake_evaluation,
        dqn_evaluation_score
    ],
    "Training Time (seconds)": [
        q_training_time,
        dqn_training_time
    ]
})

print(final_report)
