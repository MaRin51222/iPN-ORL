import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import joblib
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")



def load_and_preprocess_data(file_path, normalize=True):
    df = pd.read_csv(file_path)


    df['state'] = df['state'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['action'] = df['action'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['next_state'] = df['next_state'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['done'] = df['done'].apply(lambda x: x == 'TRUE')


    states = np.stack(df['state'].values)
    actions = np.stack(df['action'].values)
    rewards = df['reward'].values.reshape(-1, 1)
    next_states = np.stack(df['next_state'].values)
    dones = df['done'].values.reshape(-1, 1)


    state_mean, state_std = None, None
    if normalize:
        state_mean = states.mean(axis=0)
        state_std = states.std(axis=0) + 1e-8
        states = (states - state_mean) / state_std
        next_states = (next_states - state_mean) / state_std


        norm_params = {
            'state_mean': state_mean,
            'state_std': state_std
        }
        joblib.dump(norm_params, 'state_norm_params.pkl')
        print("Saved state normalization parameters to state_norm_params.pkl")

    print(f"Loaded dataset with {len(states)} transitions")
    print(f"State dimension: {states.shape[1]}, Action dimension: {actions.shape[1]}")

    return states, actions, rewards, next_states, dones, state_mean, state_std



class MicrogridDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.next_states = torch.FloatTensor(next_states)
        self.dones = torch.FloatTensor(dones)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)



class CQL_SACN:
    def __init__(self, state_dim, action_dim,
                 lr=1e-4, tau=0.01, gamma=0.995,
                 cql_weight=5.0,
                 cql_action_samples=10,
                 weight_decay=1e-5,
                 n_qs=5):

        self.n_qs = n_qs


        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())


        self.q_networks = [QNetwork(state_dim, action_dim).to(device) for _ in range(n_qs)]
        self.q_targets = [QNetwork(state_dim, action_dim).to(device) for _ in range(n_qs)]
        for q, q_target in zip(self.q_networks, self.q_targets):
            q_target.load_state_dict(q.state_dict())


        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.q_optimizers = [optim.AdamW(q.parameters(), lr=lr, weight_decay=weight_decay) for q in self.q_networks]


        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000, eta_min=1e-5
        )


        self.tau = tau
        self.gamma = gamma
        self.cql_weight = cql_weight
        self.cql_action_samples = cql_action_samples
        self.action_dim = action_dim

    def select_action(self, state, state_mean=None, state_std=None):

        if state_mean is not None and state_std is not None:
            state = (state - state_mean) / state_std

        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch


        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        batch_size = states.shape[0]
        action_dim = actions.shape[1]


        with torch.no_grad():
            next_actions = self.actor_target(next_states)

            all_target_qs = [q_target(next_states, next_actions) for q_target in self.q_targets]

            target_q = torch.min(torch.cat(all_target_qs, dim=1), dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * target_q


        total_q_loss = 0.0
        for i, (q, q_optimizer) in enumerate(zip(self.q_networks, self.q_optimizers)):

            q_current = q(states, actions)
            q_loss = nn.MSELoss()(q_current, target_q)

            dataset_actions = actions


            random_actions = torch.FloatTensor(
                batch_size, self.cql_action_samples, action_dim
            ).uniform_(-1, 1).to(device)


            with torch.no_grad():
                policy_actions = self.actor(states)
                policy_noise = 0.1 * torch.randn_like(policy_actions).to(device)
                policy_actions = (policy_actions + policy_noise).clamp(-1, 1)


            states_expanded = states.unsqueeze(1).repeat(1, self.cql_action_samples, 1)
            states_flat = states_expanded.reshape(-1, states.shape[1])
            random_actions_flat = random_actions.reshape(-1, action_dim)

            q_rand = q(states_flat, random_actions_flat).reshape(batch_size, self.cql_action_samples)
            q_dataset = q(states, dataset_actions)
            q_policy = q(states, policy_actions)
            q_all = torch.cat([q_rand, q_dataset, q_policy], dim=1)
            q_logsumexp = torch.logsumexp(q_all, dim=1, keepdim=True)
            cql_loss = (q_logsumexp - q_dataset).mean()


            total_q_loss_i = q_loss + self.cql_weight * cql_loss
            total_q_loss += total_q_loss_i


            q_optimizer.zero_grad()
            total_q_loss_i.backward(retain_graph=True if i < self.n_qs - 1 else False)
            nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            q_optimizer.step()


        current_actions = self.actor(states)
        all_q_values = [q(states, current_actions) for q in self.q_networks]
        min_q_value = torch.min(torch.cat(all_q_values, dim=1), dim=1, keepdim=True)[0]
        actor_loss = -min_q_value.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()


        for q, q_target in zip(self.q_networks, self.q_targets):
            for param, target_param in zip(q.parameters(), q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        self.lr_scheduler.step()

        return total_q_loss.item() / self.n_qs, actor_loss.item()  # 平均Q损失

    def save_model(self, path):

        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
        }
        for i, (q, q_target) in enumerate(zip(self.q_networks, self.q_targets)):
            save_dict[f'q{i}_state_dict'] = q.state_dict()
            save_dict[f'q{i}_target_state_dict'] = q_target.state_dict()

        torch.save(save_dict, path)
        print(f"Model saved to {path}")


def train_cql_sacn(dataset, epochs=1000, batch_size=1024, n_qs=5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    agent = CQL_SACN(state_dim, action_dim, n_qs=n_qs)  # 传入Q网络数量

    q_losses = []
    actor_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_q_loss = 0
        epoch_actor_loss = 0
        count = 0

        for batch in dataloader:
            q_loss, a_loss = agent.update(batch)
            epoch_q_loss += q_loss
            epoch_actor_loss += a_loss
            count += 1

        q_losses.append(epoch_q_loss / count)
        actor_losses.append(epoch_actor_loss / count)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Q Loss: {q_losses[-1]:.4f} | Actor Loss: {actor_losses[-1]:.4f}")


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(q_losses)
    plt.title("Q Loss (Total)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(actor_losses)
    plt.title("Actor Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig("training_loss_cql_sacn.png")
    plt.show()

    return agent


def main():
    seed = 42
    set_random_seed(seed)
    with open("training_seed.txt", "w") as f:
        f.write(f"Seed used: {seed}")


    file_path = r"5000.csv"
    states, actions, rewards, next_states, dones, state_mean, state_std = load_and_preprocess_data(
        file_path, normalize=True
    )

    dataset = MicrogridDataset(states, actions, rewards, next_states, dones)

    agent = train_cql_sacn(dataset, epochs=5000, batch_size=1024, n_qs=5)

    agent.save_model("microgrid.pth")


    test_state = states[1]
    print(f"Test state (normalized): {test_state[:5]}...")

    if state_mean is not None and state_std is not None:
        original_test_state = test_state * state_std + state_mean
        print(f"Original test state: {original_test_state[:5]}...")

    action = agent.select_action(test_state, state_mean, state_std)
    print(f"Predicted action ([-1,1] scale): {action}")


if __name__ == "__main__":
    main()