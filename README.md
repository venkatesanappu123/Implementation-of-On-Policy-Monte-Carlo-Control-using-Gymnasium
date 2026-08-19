# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
````

---

## Environment Description

The **FrozenLake-v1** environment is a grid-world environment provided by Gymnasium.

The environment contains four types of tiles:

* **S** → Starting state
* **F** → Frozen surface
* **H** → Hole
* **G** → Goal

The agent starts from the starting state and must reach the goal while avoiding the holes.

For this experiment, a deterministic `4 × 4` FrozenLake environment is used with `is_slippery=False`.

The environment contains:

* **16 states**
* **4 possible actions**

The action mapping is:

| Action | Meaning |
| ------ | ------- |
| 0      | Left    |
| 1      | Down    |
| 2      | Right   |
| 3      | Up      |

---

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol   | Meaning                   |
| -------- | ------------------------- |
| $s$      | Current state             |
| $a$      | Action taken in state $s$ |
| $G_t$    | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate     |
| $\alpha$ | Learning rate             |
| $\gamma$ | Discount factor           |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Import the required Gymnasium, NumPy, and Matplotlib libraries.
2. Create the `FrozenLake-v1` environment.
3. Obtain the number of states and number of actions.
4. Initialize the Q-table with zeros.
5. Set the hyperparameters such as number of episodes, learning rate, discount factor, and epsilon values.
6. Use an epsilon-greedy policy to select actions.
7. Generate a complete episode until the environment reaches a terminal state.
8. Store the state, action, and reward for every step.
9. Calculate the return $G_t$ by processing the episode backwards.
10. Update the Q-value using the Monte Carlo update rule.
11. Use first-visit Monte Carlo by updating each state-action pair only once per episode.
12. Decrease epsilon after every episode while maintaining a minimum exploration value.
13. Repeat the process for all training episodes.
14. Extract the greedy policy using the maximum Q-value for every state.
15. Calculate the estimated state-value function.
16. Display the final Q-table, state-value function, learned policy, and average reward.
17. Plot the learning curve.

---

## Python Program

---

#### Monte Carlo Control

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000

gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))

episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------


def epsilon_greedy_action(state, epsilon):
    """
    Select an action using an epsilon-greedy policy.
    """

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    # Select the action with the maximum Q-value.
    # Random tie-breaking is used when multiple
    # actions have the same Q-value.
    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))

    return np.random.choice(best_actions)


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------


def generate_episode(epsilon):
    """
    Generates one episode using the current
    epsilon-greedy policy.

    Returns a list of:
    (state, action, reward)
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate a complete episode
    episode = generate_episode(epsilon)

    # Store total reward obtained in the episode
    episode_rewards.append(sum(reward for _, _, reward in episode))

    # Initialize return
    G = 0.0

    # Keep track of visited state-action pairs
    visited = set()

    # Process the episode backwards
    for t in range(len(episode) - 1, -1, -1):

        state, action, reward = episode[t]

        # Calculate discounted return
        G = gamma * G + reward

        # First-visit Monte Carlo
        if (state, action) not in visited:

            visited.add((state, action))

            # Incremental Monte Carlo Q-value update
            Q[state, action] += alpha * (G - Q[state, action])

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Display training progress
    if (episode_num + 1) % 2000 == 0:

        avg_reward = np.mean(episode_rewards[-1000:])

        print(
            f"Episode {episode_num + 1:5d} | "
            f"epsilon = {epsilon:.3f} | "
            f"last-1000 average reward = {avg_reward:.3f}"
        )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------


def print_policy(policy):

    action_symbols = {0: "L", 1: "D", 2: "R", 3: "U"}

    policy_grid = np.array([action_symbols[action] for action in policy]).reshape(4, 4)

    print("Name: Venkatesan R")

    print("Register Number: 212224230299")

    print("\nLearned Policy:")

    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")

print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", success_rate)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")

plt.ylabel("Average Reward")

plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)

plt.show()

env.close()

```

---

## Output

### Episodes=20000:

### Final Q-table-1:

<img width="348" height="388" alt="{57B90077-4874-4AB4-BCCC-3DAA58AF9397}" src="https://github.com/user-attachments/assets/0e755255-28d9-4f50-813e-318d5acec9cb" />


### Estimated State-Value Function

<img width="380" height="178" alt="{CD250B14-3D15-4D1A-A1AF-96C49FBF693E}" src="https://github.com/user-attachments/assets/5ac2fefb-b529-4397-b804-846abd95e839" />


### Learned Policy

<img width="269" height="119" alt="{2D2B3FD2-1563-4921-B10B-C4B9EE2D5702}" src="https://github.com/user-attachments/assets/f866e60a-3421-4e97-a9cf-f21660535cf5" />


### Average Reward

<img width="468" height="38" alt="{F5C28112-739F-4FBD-BD71-3F419DD0ABC2}" src="https://github.com/user-attachments/assets/ab3e0afe-0f35-47f7-b6ab-543df2521862" />


### Plot Learning Curve:

<img width="699" height="470" alt="image" src="https://github.com/user-attachments/assets/7326cd31-9bf7-4fd9-9e1c-d46c9478d8e3" />


### Episodes=4000:

### Final Q-table-2:

<img width="389" height="385" alt="{C167E0EC-1FF5-4EB3-8C9F-EF25D6EA581D}" src="https://github.com/user-attachments/assets/090dbab1-24da-457c-9072-6d006e53fc88" />


### Estimated State-Value Function

<img width="397" height="169" alt="{CCE21BD6-90EE-4BE1-80EC-4367C046B08E}" src="https://github.com/user-attachments/assets/59b4a625-a11c-497b-a3d0-66841e46a255" />


### Learned Policy

<img width="344" height="127" alt="{81710486-DCAB-49AF-B09C-3F779257C911}" src="https://github.com/user-attachments/assets/7ac3aa87-4ae6-468c-b4de-9b695d2098fe" />


### Average Reward

<img width="484" height="31" alt="{7D446546-402F-4053-93C0-BDC0BF60EA1A}" src="https://github.com/user-attachments/assets/abb36865-ef76-4324-a10c-bff41929b4ac" />



### Plot Learning Curve:

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/4bdebd52-2ac5-40f5-94e7-0f97188f2e52" />




---

## Result

The On-Policy Monte Carlo Control algorithm was successfully implemented using Gymnasium's FrozenLake-v1 environment.
The agent learned an improved policy using Monte Carlo returns and an epsilon-greedy strategy.

---

## Inference

At the beginning, the agent explores a lot because the epsilon value is high (1.0). Over time, epsilon gradually reduces until it reaches a minimum of 0.05, so the agent starts exploring less and exploiting more.
The Monte Carlo method learns by observing complete episodes and calculating the total return, which is then used to update the Q-values.
The Q-table stores the expected rewards for each action in every state. From this, the state-value can be found by choosing the highest Q-value for each state.
The epsilon-greedy strategy helps balance exploration and exploitation—trying new actions at first, then focusing on the best actions later.
The learning curve represents how the agent’s average reward improves (or changes) as training progresses over multiple episodes.

---
---

```
```
