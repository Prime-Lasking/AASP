import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd

companies = ["AAPL", "BTC", "NVDA"]

# yfinance tickers can differ from the label you want to use in the CSV.
# Example: Bitcoin is typically available as BTC-USD.
download_tickers = {
    "AAPL": "AAPL",
    "BTC": "BTC-USD",
    "NVDA": "NVDA",
}

all_data = {}
for label in companies:
    yf_ticker = download_tickers.get(label, label)
    df = yf.download(yf_ticker, start="2020-01-01")
    if not df.empty:
        close = df[["Close"]].copy()
        close.columns = [label]
        all_data[label] = close

if not all_data:
    raise RuntimeError("No data was fetched from yfinance; cannot train models.")

combined_df = pd.concat(all_data.values(), axis=1)
combined_df.to_csv("stock.csv")

# Read CSV
df = pd.read_csv("stock.csv", index_col=0, parse_dates=True)

quick_train = os.getenv("QUICK_TRAIN", "").strip().lower() in {"1", "true", "yes", "y"}
epochs = 200 if quick_train else 5000
if quick_train:
    companies = companies[:2]

# Create weights directory if it doesn't exist
if not os.path.exists('weights'):
    os.makedirs('weights')

# Training data
# x -> guess number
# y -> right number

sequence_length = 20

for company in companies:

    if company not in df.columns:
        print(f"{company}: data not found, skipping...")
        continue

    prices = df[company].values
    prices = prices[~np.isnan(prices)]

    if len(prices) <= sequence_length:
        print(f"{company}: not enough data, skipping...")
        continue

    X = []
    y = []

    for i in range(len(prices) - sequence_length):
        X.append(prices[i : i + sequence_length])
        y.append(prices[i + sequence_length])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Training model for {company}...")
    for epoch in range(epochs):

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
        prediction = model(X_tensor)
        loss = criterion(prediction, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    safe_company = company.replace("-", "_")
    model_path = f"weights/{safe_company}_model.pth"
    torch.save(model, model_path)
    print(f"Saved {model_path} (loss: {loss.item():.6f})")