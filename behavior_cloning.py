import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        hidden_size = (input_size + output_size) // 2
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Flatten()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ExpertDataset(Dataset):
    def __init__(self, file="1e_firsttest.npz"):
        self.data =  np.load(file)
        self.obs_size = len(self.data["obs"][0])
        self.actions_size = len(self.data["actions"][0])
    
    def __len__(self):
        return len(self.data["actions"])
    
    def __getitem__(self, idx):
        obs = self.data["obs"][idx]
        action = self.data["actions"][idx]
        return obs, action

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    dataset = ExpertDataset()
    input_size = dataset.obs_size
    output_size = dataset.actions_size
    dataloader = DataLoader(dataset, batch_size=1)
    model = NeuralNetwork(input_size, output_size).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for i in range(1):
        train(dataloader, model, loss_fn, optimizer)
    
    global trained
    trained = True
    import module_env, gym
    env = gym.make(module_env.env_name)
    obs = env.reset()
    for i in range(100):
        obs = torch.from_numpy(np.array([obs])).float().to(device)
        action = model(obs)
        action = action.cpu().detach().numpy()
        obs, reward, done, _ = env.step(action[0])
        print(action)



