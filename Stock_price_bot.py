import telebot
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Your Alpha Vantage API key
ALPHAVANTAGE_API_KEY = 'WXDR7I7WJTZ1P5TW'

# =========================
# Function to fetch real-time and historical data from Alpha Vantage
# =========================
def get_stock_data(symbol):
    try:
        # Fetch real-time quote
        url_quote = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}'
        response_quote = requests.get(url_quote)
        data_quote = response_quote.json()
        current_price = None
        if 'Global Quote' in data_quote and '05. price' in data_quote['Global Quote']:
            current_price = float(data_quote['Global Quote']['05. price'])
        else:
            return None, None

        # Fetch historical data (full daily data)
        url_history = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHAVANTAGE_API_KEY}'
        response_history = requests.get(url_history)
        data_history = response_history.json()

        if 'Time Series (Daily)' not in data_history:
            return None, current_price

        time_series = data_history['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })

        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if len(df) < 30:
            return None, current_price

        return df, current_price
    except:
        return None, None

# =========================
# Analysis Class
# =========================
class AdvancedStockAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lookback_days = 10

    def calculate_technical_indicators(self, df):
        df = df.copy()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=5).std()
        df['volume_sma'] = df['volume'].rolling(window=5).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        return df.dropna()

    def prepare_features(self, df):
        df = self.calculate_technical_indicators(df)
        features = []
        targets = []

        for i in range(self.lookback_days, len(df) - 1):
            recent_prices = df['close'].iloc[i - self.lookback_days:i]
            recent_volumes = df['volume'].iloc[i - self.lookback_days:i]
            feature_set = [
                recent_prices.pct_change().mean(),
                recent_prices.pct_change().std(),
                recent_volumes.pct_change().mean(),
                df.iloc[i]['price_vs_sma5'],
                df.iloc[i]['price_vs_sma10'],
                df.iloc[i]['volatility'],
                df.iloc[i]['daily_range'],
                df.iloc[i]['close'],
                i % 7,
                len(df) - i
            ]
            target_price = df.iloc[i + 1]['close']
            features.append(feature_set)
            targets.append(target_price)
        return np.array(features), np.array(targets)

    def train_and_predict(self, df):
        try:
            features, targets = self.prepare_features(df)
            if len(features) < 15:
                return None, "Not enough data"
            split_idx = int(0.8 * len(features))
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            accuracy = mean_absolute_error(y_test, predictions)
            latest_features = features[-1].reshape(1, -1)
            tomorrow_prediction = self.model.predict(latest_features)[0]
            current_price = df['close'].iloc[-1]
            predicted_change = ((tomorrow_prediction - current_price) / current_price) * 100
            result = {
                'current_price': float(round(current_price, 2)),
                'predicted_price': float(round(tomorrow_prediction, 2)),
                'predicted_change_percent': float(round(predicted_change, 2)),
                'model_accuracy': float(round(accuracy, 2)),
                'direction': 'UP' if predicted_change > 0 else 'DOWN',
                'confidence_score': max(1, min(10, 10 - (accuracy / current_price * 100)))
            }
            return result, "Success"
        except:
            return None, "Error during prediction"

    def analyze_trend(self, df):
        short_term = df['close'].tail(5)
        short_change = ((short_term.iloc[-1] - short_term.iloc[0]) / short_term.iloc[0]) * 100
        medium_term = df['close'].tail(20)
        medium_change = ((medium_term.iloc[-1] - medium_term.iloc[0]) / medium_term.iloc[0]) * 100
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        if medium_change > 10:
            trend = "STRONG UPTREND 📈"
        elif medium_change > 3:
            trend = "UPTREND ↗️"
        elif medium_change > -3:
            trend = "SIDEWAYS ↔️"
        elif medium_change > -10:
            trend = "DOWNTREND ↘️"
        else:
            trend = "STRONG DOWNTREND 📉"
        return {
            'trend': trend,
            'short_term_change': float(round(short_change, 2)),
            'medium_term_change': float(round(medium_change, 2)),
            'volume_status': 'HIGH' if volume_ratio > 1.2 else 'NORMAL',
            'volume_ratio': float(round(volume_ratio, 2))
        }

    def get_trading_recommendation(self, prediction_data, trend_data):
        pred_change = prediction_data['predicted_change_percent']
        confidence = prediction_data['confidence_score']
        trend = trend_data['trend']
        if pred_change > 3 and confidence > 7 and "UPTREND" in trend:
            action = "STRONG BUY 🟢"
            reason = "Strong upward momentum with high confidence"
        elif pred_change > 1.5 and confidence > 6:
            action = "BUY 🟢"
            reason = "Positive prediction with good confidence"
        elif pred_change < -3 and confidence > 7 and "DOWNTREND" in trend:
            action = "STRONG SELL 🔴"
            reason = "Strong downward momentum with high confidence"
        elif pred_change < -1.5 and confidence > 6:
            action = "SELL 🔴"
            reason = "Negative prediction with good confidence"
        elif abs(pred_change) < 0.5:
            action = "HOLD ⚪"
            reason = "Minimal expected movement"
        else:
            action = "HOLD 🟡"
            reason = "Mixed signals - wait for clearer direction"
        return action, int(confidence), reason

# =========================
# Wrapper function to analyze stock data
# =========================
def analyze_stock(data, symbol):
    """
    Wraps the analysis process, returns formatted report.
    """
    try:
        analyzer = AdvancedStockAnalyzer()
        prediction, status = analyzer.train_and_predict(data)
        if prediction is None:
            return f"❌ Analysis failed: {status}"

        trend = analyzer.analyze_trend(data)
        action, confidence, reason = analyzer.get_trading_recommendation(prediction, trend)

        result = f"""
🤖 **STOCK ANALYSIS REPORT**
**Symbol:** {symbol}

🎯 **TRADING SIGNAL**
**Action:** {action}
**Confidence:** {confidence}/10
**Reason:** {reason}

💰 **PRICE ANALYSIS**
• Current Price: ${prediction['current_price']}
• Predicted Price: ${prediction['predicted_price']}
• Expected Change: {prediction['predicted_change_percent']}% ({prediction['direction']})
• Model MAE (approximate): ${prediction['model_accuracy']}

📈 **TREND ANALYSIS**
• Market Trend: {trend['trend']}
• 5-day Performance: {trend['short_term_change']}%
• 20-day Performance: {trend['medium_term_change']}%
• Volume: {trend['volume_status']} ({(trend['volume_ratio']-1)*100:.1f}% vs avg)

⚠️ **DISCLAIMER**
This is for educational purposes only.
Always do your own research before investing.

⏰ **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return result
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"

# =========================
# Telegram Bot Setup
# =========================
API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN_HERE'  # <-- Replace with your actual token
bot = telebot.TeleBot(API_TOKEN)

AVAILABLE_STOCKS = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet (Google)',
    'MSFT': 'Microsoft',
    'TSLA': 'Tesla',
    'AMZN': 'Amazon',
    'META': 'Meta Platforms',
    'NFLX': 'Netflix',
    'NVDA': 'NVIDIA',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson'
}

# =========================
# Bot Handlers
# =========================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 **Welcome to Advanced Stock Prediction Bot!**

I analyze stocks and give buy/sell recommendations using machine learning and real-time data.

**Commands:**
/start /help - Show this message
/stocks - List available stocks
/analyze [SYMBOL] - Analyze specific stock
/quick [NUMBER] - Quick analyze from list

**Examples:**
/analyze AAPL
/quick 1 (Apple)
/TSLA (send just symbol)

**Features:**
✅ Current & Predicted Prices
✅ BUY/SELL/HOLD with Confidence
✅ Trend & Volume Analysis
"""
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['stocks'])
def list_stocks(message):
    text = "📊 **Available Stocks:**\n\n"
    for i, (symbol, name) in enumerate(AVAILABLE_STOCKS.items(), 1):
        text += f"{i}. {symbol} - {name}\n"
    text += "\nUse `/quick 1` for Apple, or `/analyze SYMBOL` for any stock."
    bot.reply_to(message, text, parse_mode='Markdown')

@bot.message_handler(commands=['quick'])
def quick_analyze(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "Please specify a number. Example: /quick 1")
            return
        num = int(parts[1])
        symbols = list(AVAILABLE_STOCKS.keys())
        if 1 <= num <= len(symbols):
            symbol = symbols[num - 1]
            bot.send_message(message.chat.id, f"🔍 Analyzing {symbol}...")
            data, current_price = get_stock_data(symbol)
            if data is None:
                bot.send_message(message.chat.id, f"❌ Unable to retrieve data for {symbol} at this time.")
                return
            result = analyze_stock(data, symbol)
            bot.send_message(message.chat.id, result, parse_mode='Markdown')
        else:
            bot.reply_to(message, f"Number must be between 1 and {len(symbols)}.")
    except ValueError:
        bot.reply_to(message, "Please provide a valid number.")
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")

@bot.message_handler(commands=['analyze'])
def analyze_command(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "Please specify a stock symbol. Example: /analyze AAPL")
            return
        symbol = parts[1].upper()
        bot.send_message(message.chat.id, f"🔍 Analyzing {symbol}...")
        data, current_price = get_stock_data(symbol)
        if data is None:
            bot.send_message(message.chat.id, f"❌ Unable to retrieve data for {symbol} at this time.")
            return
        result = analyze_stock(data, symbol)
        bot.send_message(message.chat.id, result, parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")

@bot.message_handler(func=lambda msg: True)
def handle_symbol_message(message):
    text = message.text.strip().upper()
    if text in AVAILABLE_STOCKS or (text.isalpha() and 2 <= len(text) <= 5):
        bot.send_message(message.chat.id, f"🔍 Analyzing {text}...")
        data, current_price = get_stock_data(text)
        if data is None:
            bot.send_message(message.chat.id, f"❌ Unable to retrieve data for {text} at this time.")
            return
        result = analyze_stock(data, text)
        bot.send_message(message.chat.id, result, parse_mode='Markdown')
    else:
        bot.reply_to(message, """
I didn't understand that. You can:
• Send a stock symbol like AAPL
• Use /stocks to see list
• Use /quick 1 for Apple
• Use /analyze SYMBOL for any stock
• Or just send the symbol directly.
""")

def main():
    print("🤖 Stock Prediction Bot Started!")
    print("Bot is running... Press Ctrl+C to stop.")
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"Bot error: {e}")
        time.sleep(5)
        main()

if __name__ == '__main__':
    main()
