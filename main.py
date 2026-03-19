import pandas as pd
import os
from engine import EthereumTradingEngine

def run_backtest(csv_path, initial_funds=100000.0):
    # 1. Clean up old logs so we don't append to old data
    if os.path.exists('logs.csv'):
        os.remove('logs.csv')

    # 2. Load Data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Ensure your CSV columns match these names (timestamp, close)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Initialize Engine
    engine = EthereumTradingEngine(initial_balance=initial_funds, fee_rate=0.001)

    print(f"Starting Backtest...")

    # 4. Strategy Loop
    for i in range(len(df)):
        row = df.iloc[i]
        
        # --- PLACEHOLDER STRATEGY ---
        # Replace this with your actual logic
        # Example: Just a dummy signal for testing
        # 1 = Long, -1 = Short, 0 = Do nothing
        test_signal = 0 
        if i == 10: test_signal = 1   # Open Long at 10th hour
        if i == 20: test_signal = -1  # Close Long & Open Short at 20th hour
        if i == 30: test_signal = 1   # Close Short & Open Long at 30th hour
        
        # Use 10% of current balance as budget for each trade
        current_budget = engine.balance * 0.1
        
        engine.execute_trade(
            signal=test_signal,
            budget=current_budget,
            market_price=row['close'],
            timestamp=row['timestamp'],
            gas_fee=0.0 # Your gas cost in RS/USD
        )

    # 5. Final Summary
    print("\n" + "="*30)
    print(f"Backtest Finished.")
    print(f"Final Account Balance: {engine.balance:.2f}")
    print("Check 'logs.csv' for the full trade history.")
    print("="*30)

if __name__ == "__main__":
    # Point this to your actual 2022-2026 Ethereum CSV file
    DATA_FILE = 'ETH-USDT_1h.csv' 
    run_backtest(DATA_FILE)