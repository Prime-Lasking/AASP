import numpy as np
import pandas as pd
import torch

import os

companies = ["AAPL", "BTC", "NVDA"]

df = pd.read_csv("stock.csv", index_col=0, parse_dates=True)

# Use the same sequence length as in training
sequence_length = 20

# Prepare the input sequence (last 'sequence_length' prices)
for company in companies:
    safe_company = company.replace("-", "_")
    model_path = f"weights/{safe_company}_model.pth"
    if not os.path.exists(model_path):
        print(f"{company}: model not found, skipping...")
        continue

    if company not in df.columns:
        print(f"{company}: data not found, skipping...")
        continue

    prices = df[company].values
    prices = prices[~np.isnan(prices)]
    if len(prices) < sequence_length:
        print(f"{company}: not enough data, skipping...")
        continue

    model = torch.load(model_path, weights_only=False)
    model.eval()

    last_sequence = prices[-sequence_length:]
    last_sequence = np.array(last_sequence)
    input_tensor = torch.tensor(last_sequence, dtype=torch.float32)

    with torch.no_grad():
        predicted_next = model(input_tensor)

    print(f"{company} current: {prices[-1]:.2f} | predicted next: {predicted_next.item():.2f}")
