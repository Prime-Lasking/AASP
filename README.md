# Apple Stock Predictor (AASP)

A machine learning project that predicts Apple Inc. (AAPL) stock prices using a neural network implemented in PyTorch. This project demonstrates time series forecasting for stock market data.

## Features

- Fetches real-time Apple stock data using Yahoo Finance
- Implements a deep neural network for price prediction
- Uses historical price data to forecast future stock prices
- Configurable sequence length for time-series analysis
- Training progress visualization

## Requirements

- Python 3.6+
- PyTorch
- pandas
- yfinance
- scikit-learn (for data preprocessing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Prime-Lasking/AASP.git
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

1. First, fetch the latest stock data:
   ```bash
   python stock.py
   ```
   This will download the latest Apple stock data and save it as `stock.csv`.

2. Run the prediction model:
   ```bash
   python main.py
   ```
   The script will train the neural network and output the predicted next closing price.

## Model Architecture

The neural network consists of:
- Input layer (sequence length)
- 4 Hidden layers (32 neurons each) with ReLU activation
- Output layer (1 neuron for price prediction)

## Training

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer with a learning rate of 0.01
- 10,000 training epochs
- Sequence length of 20 days for time-series analysis

## Results

The model outputs the predicted next day's closing price based on the most recent sequence of closing prices. The prediction is printed at the end of the training process.

## Files

- `main.py`: Main script containing the neural network implementation and training loop
- `stock.py`: Script to fetch stock data using yfinance
- `stock.csv`: CSV file containing historical stock data
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
