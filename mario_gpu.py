import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from torchrl.data import TensorDictReplayBuffer, ListStorage
from tensordict import TensorDict
from agent_nn import AgentNN  # Your NN definition goes here
import os

# -------------------------------
# Define wrappers for the environment
# -------------------------------


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def apply_wrappers(env):
    env = SkipFrame(env, skip=5)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (40, 40))
    env = FrameStack(env, num_stack=5, lz4_compress=True)
    return env

# -------------------------------
# Define your Agent with saving and loading functions
# -------------------------------


class Agent:
    def __init__(
        self,
        input_dims,
        num_actions,
        lr=0.00025,
        gamma=0.9,
        epsilon=1.0,
        eps_decay=0.9999995,
        eps_min=0.011,
        replay_buffer_capacity=100000,  # Reduced for Windows stability
        batch_size=32,
        sync_network_rate=10000
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Create the networks and move them to the device
        self.online_network = AgentNN(input_dims, num_actions).to(self.device)
        self.target_network = AgentNN(
            input_dims, num_actions, freeze=True).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Use RAM-based storage to avoid mmap errors
        storage = ListStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def preprocess_obs(self, obs):
        # Convert obs to a torch tensor on the correct device
        obs = torch.tensor(
            np.array(obs), dtype=torch.float32, device=self.device)
        # If obs is 4D with a final channel of 1 (grayscale), squeeze that dimension
        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = obs.squeeze(-1)
        return obs

    def choose_action(self, observation):
        # Epsilon-greedy strategy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Convert observation to correct shape and device
        observation = self.preprocess_obs(observation).unsqueeze(0)
        with torch.no_grad():
            return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        state = self.preprocess_obs(state)
        next_state = self.preprocess_obs(next_state)

        self.replay_buffer.add(
            TensorDict(
                {
                    "state": state,
                    "action": torch.tensor(action, device=self.device),
                    "reward": torch.tensor(reward, device=self.device),
                    "next_state": next_state,
                    "done": torch.tensor(done, device=self.device),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.online_network.load_state_dict(
                torch.load(path, map_location=self.device))
            self.target_network.load_state_dict(
                torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()

        self.optimizer.zero_grad()
        samples = self.replay_buffer.sample(self.batch_size).to(self.device)

        states, actions, rewards, next_states, dones = [
            samples[k] for k in ("state", "action", "reward", "next_state", "done")
        ]

        # Q(s) for online network
        q_pred = self.online_network(states)
        # Choose the Q-values corresponding to the actions that were taken
        q_pred = q_pred[torch.arange(self.batch_size), actions.squeeze()]

        # Q(s') for target network
        q_next = self.target_network(next_states).max(dim=1)[0]

        # Q-target for the loss
        q_target = rewards + self.gamma * q_next * (1 - dones.float())

        loss = self.loss(q_pred, q_target)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


# -------------------------------
# Environment and Training Setup
# -------------------------------
ENV_NAME = 'SuperMarioBros-v0'
env = gym_super_mario_bros.make(
    ENV_NAME, apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape,
              num_actions=env.action_space.n)

# Optionally load an existing model (uncomment the next line if desired)
agent.load_model("agent_model_latest.pth")

RENDER_EVERY = 5000   # Render every 5,000 episodes (if desired)
PRINT_EVERY = 500     # Print progress every 500 episodes
SAVE_EVERY = 100     # Save the model every 5,000 episodes

NUM_EPISODES = 50000

for episode in range(NUM_EPISODES):
    render = (episode % RENDER_EVERY == 0)
    print_progress = (episode % PRINT_EVERY == 0)

    done = False
    state, _ = env.reset()
    total_reward = 0
    step = 0

    while not done:
        if render:
            env.render()

        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)

        agent.store_in_memory(state, action, reward,
                              new_state, done or truncated)

        if step % 4 == 0:  # Learn every 4 steps
            agent.learn()

        state = new_state
        total_reward += reward
        step += 1

    if print_progress:
        print(
            f"[Episode {episode}] Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    # Save the model periodically
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save_model("agent_model_latest.pth")

# Save one last time after training completes
agent.save_model("agent_model_latest.pth")
env.close()
