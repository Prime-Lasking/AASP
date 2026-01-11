import torch
import pandas as pd
import numpy as np

# Load the trained model with weights_only=False for compatibility
model = torch.load('weights/model.pth', weights_only=False)
model.eval()  # Set the model to evaluation mode

# Load the stock data
df = pd.read_csv("stock.csv")
prices = df["Close"].values

# Use the same sequence length as in training
sequence_length = 20

# Prepare the input sequence (last 'sequence_length' prices)
last_sequence = prices[-sequence_length:]
last_sequence = np.array(last_sequence)  # Convert to numpy array
input_tensor = torch.tensor(last_sequence, dtype=torch.float32)

# Make prediction
with torch.no_grad():
    predicted_next = model(input_tensor)

print(f"Last {sequence_length} prices: {last_sequence}")
print(f"Predicted next price: {predicted_next.item():.2f}")
