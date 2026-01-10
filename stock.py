import yfinance as yf

df = yf.download("aapl", start="2025-01-01")
df.to_csv("stock.csv")
