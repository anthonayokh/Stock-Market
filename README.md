# Stock-Market
Predict Stock Market Price
# Stock Price Prediction Bot - Python

[![Python Version](https://img.shields.io/badge/Python-3.9%20or%20higher-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/your_username/your_repo.svg)](LICENSE)
[![Commit Status](https://github.com/your_username/your_repo/commits/main)](https://github.com/your_username/your_repo)

## Description

This project creates a Telegram bot that predicts stock prices using historical data fetched from Alpha Vantage and a Random Forest Regression model. The bot provides real-time price updates and predictions based on specified stock symbols.

## Prerequisites

*   **Python 3.7+:**  [https://www.python.org/downloads/](https://www.python.org/downloads/)
*   **Telegram Bot:**  You'll need to create a Telegram bot using BotFather and obtain its token.
*   **Alpha Vantage API Key:** You'll need to sign up for an Alpha Vantage account and obtain an API key.
*   **Libraries:** Install the necessary Python libraries using pip:

    ```bash
    pip install telebot pandas numpy scikit-learn requests
    ```

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [your_github_url]
    cd your_repo
    ```

2.  **Set API Key:**  Replace `"WXDR7I7WJTZ1P5TW"` with your actual Alpha Vantage API key.  This is done in the `get_stock_data()` function.

3.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```

## Usage

1.  **Start the Bot:**  Find your bot's username on Telegram and start a chat with it.
2.  **Send a Command:** Send a message to the bot with the stock symbol you want to track (e.g., `AAPL`, `MSFT`, `GOOG`).
3.  **View Prediction:** The bot will provide the current price and a predicted price based on the model.

## Code Structure

*   `stock_prediction_bot.py`: Main script containing the bot logic, data fetching, and prediction.
*   `get_stock_data.py`:  Helper functions for fetching data from Alpha Vantage.
*   `model.py`: Contains the Random Forest model and prediction functions.

##  Dependencies

*   **Telebot:**  For creating and interacting with Telegram bots.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Scikit-learn:** For building and evaluating the Random Forest model.
*   **Requests:** For making HTTP requests to the Alpha Vantage API.

##  Contributing

Feel free to contribute to this project by:

*   Reporting bugs
*   Suggesting improvements
*   Adding new features

##  License

[MIT License](LICENSE)

---
