# Multi-Ticker Price Predictor (MTPP)

A machine learning project that predicts next-step closing prices for multiple tickers using neural networks implemented in PyTorch. This project demonstrates time series forecasting for market data.

## Features

- Fetches market data using Yahoo Finance (download happens inside `train.py`)
- Trains a separate deep neural network per ticker
- Uses historical closing prices to forecast the next closing price
- Configurable sequence length for time-series analysis
- Training progress visualization

## Tickers

The current default tickers are:
- `AAPL` (Apple)
- `BTC` (Bitcoin, downloaded as `BTC-USD` via yfinance)
- `NVDA` (NVIDIA)

## Requirements

- Python 3.6+
- PyTorch
- pandas
- yfinance
- numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Prime-Lasking/MTPP.git
   cd AASP
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install them individually:
   ```bash
   pip install torch pandas yfinance scikit-learn
   ```

## Usage

1. Train the models (this step also downloads the latest data and saves `stock.csv` locally):
   ```bash
   python train.py
   ```
   Optional quick test run:
   ```bash
   QUICK_TRAIN=1 python train.py
   ```

2. Run predictions:
   ```bash
   python main.py
   ```
   The script loads each trained model from `weights/` and prints the current and predicted next price.

## Model Architecture

The neural network consists of:
- Input layer (sequence length)
- 4 Hidden layers (32 neurons each) with ReLU activation
- Output layer (1 neuron for price prediction)

## Training

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with a learning rate of 0.01
- 5,000 training epochs (default)
- Sequence length of 20 days for time-series analysis

## Results

The model outputs the predicted next day's closing price based on the most recent sequence of closing prices. The prediction is printed at the end of the training process.

## Files

- `train.py`: Downloads data (yfinance), prepares training sequences, trains models, and saves weights to `weights/`
- `main.py`: Loads trained models and prints next-step predictions per ticker
- `stock.csv`: Generated dataset used for training (gitignored)
- `weights/`: Generated trained models (gitignored)
- `README.md`: This file

## Note on Predictions

Stock market prediction is inherently uncertain and past performance is not indicative of future results. This project is for educational purposes only and should not be used for making investment decisions.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for providing stock market data
- PyTorch team for the deep learning framework
- Pandas for data manipulation
