import math
import numpy as np
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

# --- Configuration ---
STOCKS_TO_TEST = ["SPY", "SSO", "UPRO", "TQQQ"]
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
CASH = 10000
COMMISSION = 0.000  # Alpaca has 0 commissions for standard equities

# --- Indicator Helper Functions for Backtesting.py ---
def calc_sma(close, length):
    return ta.sma(pd.Series(close), length=length).to_numpy()

def calc_rsi(close, length):
    return ta.rsi(pd.Series(close), length=length).to_numpy()

def calc_atr(high, low, close, length):
    return ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=length).to_numpy()

def calc_bbu(close, length, std):
    bb = ta.bbands(pd.Series(close), length=length, std=std)
    # The upper band is the 3rd column (index 2)
    return bb.iloc[:, 2].to_numpy()

class InstitutionalDip(Strategy):
    # --- Strategy Settings (Matched to live_market.py) ---
    sma_period = 100       
    rsi_period = 14
    rsi_lower = 55      # Lowered to 40 for a truer dip
    rsi_upper = 70
    bb_length = 20
    bb_std = 2.0
    atr_period = 14
    atr_multiplier = 2.5   
    reward_ratio = 99.0     # Added Take Profit Ratio
    risk_per_trade = 0.02  # 2% Account Risk

    def init(self):
        # 1. Trend Filter
        self.sma = self.I(calc_sma, self.data.Close, self.sma_period)
        
        # 2. Momentum
        self.rsi = self.I(calc_rsi, self.data.Close, self.rsi_period)
        
        # 3. Volatility (Upper Band for Early Exit)
        self.bb_upper = self.I(calc_bbu, self.data.Close, self.bb_length, self.bb_std)

        # 4. Risk (ATR)
        self.atr = self.I(calc_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        # Skip evaluating until indicators have enough data to populate
        if pd.isna(self.sma[-1]) or pd.isna(self.atr[-1]):
            return
            
        price = float(self.data.Close[-1])
        
        # --- DYNAMIC EXIT LOGIC ---
        if self.position:
            # Sell early if momentum shifts (RSI > 70 or hits Upper BB)
            # (If this doesn't hit, the hard SL or TP attached to the buy order will catch it)
            if self.rsi[-1] > self.rsi_upper or price > self.bb_upper[-1]:
                self.position.close()
                return

        # --- ENTRY LOGIC ---
        is_uptrend = price > self.sma[-1]
        is_oversold = self.rsi[-1] < self.rsi_lower
        
        if not self.position and is_uptrend and is_oversold:
            # 1. Calculate Bracket Prices
            risk_distance = self.atr_multiplier * self.atr[-1]
            stop_price = price - risk_distance
            take_profit_price = price + (risk_distance * self.reward_ratio)
            
            # 2. Institutional Position Sizing (Risk 2% of Equity)
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = price - stop_price
            
            if risk_per_share > 0:  
                
                invest_amount = self.equity * 0.95
                qty = math.floor(invest_amount / price)
                    
                if qty > 0:
                    self.buy(size=qty, sl=stop_price)
                '''
                # Calculate how many shares we can buy without blowing 2% risk
                qty = math.floor(risk_amount / risk_per_share)
                
                # Safety check: Don't exceed actual buying power
                max_qty = math.floor(self.equity / price)
                qty = min(qty, max_qty)
                
                if qty > 0:
                    # Execute with strict Bracket Order
                    self.buy(size=qty, sl=stop_price, tp=take_profit_price)
                '''
def run_test(symbol):
    print(f"Testing {symbol}...", end=" ", flush=True)
    
    try:
        # Download historical data
        data = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
        
        if data.empty:
            print("Failed: No data.")
            return None

        # Clean yfinance MultiIndex formatting
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Standardize column names for the backtester
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.index.name = 'Date'
        
        # Run Backtest
        bt = Backtest(data, InstitutionalDip, cash=CASH, commission=COMMISSION, exclusive_orders=True)
        stats = bt.run()
        print("Done.")
        
        return {
            "Ticker": symbol,
            "Bot Return %": round(stats['Return [%]'], 2),
            "Buy & Hold %": round(stats['Buy & Hold Return [%]'], 2),
            "Max Drawdown %": round(stats['Max. Drawdown [%]'], 2),
            "Win Rate %": round(stats['Win Rate [%]'], 2), # Added Win Rate
            "# Trades": stats['# Trades']
        }
    except Exception as e:
        print(f"Failed: {e}")
        return None

if __name__ == "__main__":
    results = []
    print("\n--- LIVE MARKET STRATEGY BACKTEST ---\n")
    
    for stock in STOCKS_TO_TEST:
        res = run_test(stock)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        
        # Extract the Buy & Hold return of SPY as the ultimate benchmark
        spy_benchmark = df.loc[df['Ticker'] == 'SPY', 'Buy & Hold %'].values[0]
        df["Beat SPY?"] = df["Bot Return %"] > spy_benchmark
        
        print("\n--- FINAL RESULTS ---")
        print(f"Target to Beat (SPY Buy & Hold): {spy_benchmark}%")
        print(df.to_string(index=False))