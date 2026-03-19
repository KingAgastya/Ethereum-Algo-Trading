import pandas as pd
import os
from engine import EthereumTradingEngine

def run_backtest(csv_path, initial_funds=100000.0):
    # 1. Clean up old logs
    if os.path.exists('logs.csv'):
        os.remove('logs.csv')

    # 2. Load and Prepare Data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Ensure timestamp is datetime and data is sorted
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 3. Calculate Strategy Indicators (SMA 20 and SMA 50)
    # Since these are hourly timestamps, 'window=20' means 20 hours.
    # For a true "20 Day" SMA on hourly data, you would use 20 * 24.
    # I will use 20 and 50 as requested, but feel free to multiply by 24.
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # 4. Initialize Engine
    engine = EthereumTradingEngine(initial_balance=initial_funds, fee_rate=0.001)

    print(f"Starting SMA Crossover Backtest (20/50)...")

    # 5. The "No-Bias" Loop
    # We loop until len(df) - 1 because we need the 'next' row for the trade price
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Skip if SMAs aren't calculated yet
        if pd.isna(current_row['sma_50']):
            continue

        # --- STRATEGY LOGIC ---
        # Signal 1: Fast SMA > Slow SMA (Bullish)
        # Signal -1: Fast SMA < Slow SMA (Bearish)
        
        if current_row['sma_20'] > current_row['sma_50']:
            target_signal = 1
        else:
            target_signal = -1

        # Use 20% of current liquid balance as the budget for the trade
        current_budget = engine.balance * 0.20

        # --- EXECUTION ---
        # We use the NEXT row's Open price to avoid look-forward bias.
        # This represents the price at the very start of the next hour.
        engine.execute_trade(
            signal=target_signal,
            budget=current_budget,
            market_price=next_row['open'], 
            timestamp=next_row['timestamp'],
            gas_fee=0.0  # Gas is 0 as per your instructions
        )

    # 7. Close any remaining open position at the final recorded price
    if engine.current_position is not None:
        last_row = df.iloc[-1]
        print(f"[{last_row['timestamp']}] Finalizing Backtest: Closing remaining open position...")
        
        # We use the 'close' of the very last row as the exit price
        engine.execute_trade(
            signal=-1 if engine.current_position['type'] == 'long' else 1,
            budget=0, # Budget doesn't matter for closing
            market_price=last_row['close'],
            timestamp=last_row['timestamp'],
            gas_fee=0.0
        )

    # 6. Final Results Summary
    print("\n" + "="*40)
    print("BACKTEST RESULTS")
    print(f"Final Balance: {engine.balance:.2f} RS")
    
    # Calculate simple benchmark (Buy and Hold)
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    bh_return = ((end_price - start_price) / start_price) * 100
    print(f"Buy & Hold Return: {bh_return:.2f}%")
    
    history = engine.get_logs()
    if not history.empty:
        win_rate = (history['pnl%'] > 0).sum() / len(history) * 100
        print(f"Strategy Win Rate: {win_rate:.2f}%")
        print(f"Total Trades Executed: {len(history)}")
    print("="*40)



if __name__ == "__main__":
    # Ensure this matches your filename
    DATA_FILE = 'ETH-USDT_1h.csv' 
    run_backtest(DATA_FILE)