import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import discrete_module_env
import module_env
import gym
from scipy.special import softmax


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
    def __init__(self, env):
        import tkinter
        import tkinter.filedialog as filedialog
        tkinter.Tk().withdraw()
        self.data =  [np.load(f, allow_pickle=True) for f in filedialog.askopenfilenames()]
        merged_data = {}
        for data in self.data:
            for k,v in data.items():
                try:
                    merged_data[k] = np.append(merged_data[k], v, axis=0)
                except KeyError:
                    merged_data.update({k:v})
        self.data = merged_data
        self.obs_size = len(self.data["obs"][0])
        self.actions_size = env.action_size
    
    def __len__(self):
        return len(self.data["actions"])
    
    def __getitem__(self, idx):
        obs = self.data["obs"][idx]
        action = self.data["actions"][idx]
        return obs, action

def train(dataloader, model, loss_fn):
    num_epochs = 5
    num_points = len(dataloader.dataset) 
    size = num_points * num_epochs
    # model.train()
    rate = 5e-3
    for epoch in range(num_epochs):
        print(f"LR: {rate}")
        optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9)
        rate = rate * 0.99
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)

            # zero the paramete gradients
            optimizer.zero_grad()

            # Compute prediction error
            pred = model(X.float())
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 25 == 0:
                loss, current = loss.item(), batch * len(X)
                current = current + num_points * epoch 
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    env = gym.make(discrete_module_env.env_name)
    dataset = ExpertDataset(env)
    input_size = dataset.obs_size
    output_size = dataset.actions_size
    dataloader = DataLoader(dataset, batch_size=3)
    model = NeuralNetwork(input_size, output_size).to(device)

    loss_fn = nn.CrossEntropyLoss()
    train(dataloader, model, loss_fn)
    
    global trained
    trained = True
    obs = env.reset()
    print(obs)
    for j in range(5):
        for i in range(500):
            obs = torch.from_numpy(np.array([obs])).float().to(device)
            action = model(obs)
            action = softmax(action.cpu().detach().numpy())
            action = np.random.choice(range(output_size), p=action[0])
            if np.random.rand() < 0.2:
                action = np.random.choice(range(env.action_size))
            obs, reward, done, _ = env.step(action)
            print(action)
        obs = env.reset()



