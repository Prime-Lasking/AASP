import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Read CSV
df = pd.read_csv("stock.csv")

# Take only the closing prices
prices = df["Close"].values  # numpy array: [101, 103, 106, 110, 160, ...]

# Training data
# x -> guess number
# y -> right number

sequence_length = 20
X = []
y = []

for i in range(len(prices) - sequence_length):
    X.append(prices[i : i + sequence_length])
    y.append(prices[i + sequence_length])

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # shape (N,1)

# Neural network
model = nn.Sequential(
    nn.Linear(sequence_length, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(10000):
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}")
    prediction = model(X_tensor)
    loss = criterion(prediction, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model, 'model.pth')