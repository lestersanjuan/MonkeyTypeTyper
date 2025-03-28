from torchrl.data import TensorDictReplayBuffer, ListStorage
from tensordict import TensorDict
from agent_nn import AgentNN
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


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


class Agent:
    def __init__(self,
                 input_dims,
                 num_actions,
                 lr=0.00025,
                 gamma=0.9,
                 epsilon=1.0,
                 eps_decay=0.99999975,
                 eps_min=0.1,
                 replay_buffer_capacity=10_000,  # Reduced for Windows stability
                 batch_size=32,
                 sync_network_rate=10000):

        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Use RAM-based storage to avoid mmap errors
        storage = ListStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def preprocess_obs(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = obs.squeeze(-1)
        return obs

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        observation = self.preprocess_obs(observation).unsqueeze(
            0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        state = self.preprocess_obs(state)
        next_state = self.preprocess_obs(next_state)

        self.replay_buffer.add(TensorDict({
            "state": state,
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": next_state,
            "done": torch.tensor(done)
        }, batch_size=[]))

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(
            self.batch_size).to(self.online_network.device)
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [
            samples[k] for k in keys]

        q_pred = self.online_network(states)
        q_pred = q_pred[np.arange(self.batch_size), actions.squeeze()]

        q_next = self.target_network(next_states).max(dim=1)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones.float())

        loss = self.loss(q_pred, q_target)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


ENV_NAME = 'SuperMarioBros-v2'
env = gym_super_mario_bros.make(
    ENV_NAME, apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape,
              num_actions=env.action_space.n)

RENDER_EVERY = 10000  # Only render every N episodes
PRINT_EVERY = 500    # Only print progress every N episodes

for episode in range(50000):
    render = episode % RENDER_EVERY == 0
    print_progress = episode % PRINT_EVERY == 0

    done = False
    state, _ = env.reset()

    total_reward = 0  # For monitoring performance

    while not done:

        env.render()

        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)

        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()

        state = new_state
        total_reward += reward

    if print_progress:
        print(
            f"[Episode {episode}] Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

env.close()
