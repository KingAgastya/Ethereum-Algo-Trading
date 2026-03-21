import pandas as pd
import numpy as np
import os
from engine import EthereumTradingEngine
import matplotlib.pyplot as plt

def run_backtest(csv_path, initial_funds=100000.0):
    # 1. Clean up old logs
    if os.path.exists('logs.csv'):
        os.remove('logs.csv')

    # 2. Load and Prepare Data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 4. Initialize Engine
    engine = EthereumTradingEngine(initial_balance=initial_funds, fee_rate=0.001)

    # 5. The "No-Bias" Loop
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        # Get the current date from the timestamp
        curr_date = current_row['timestamp']
        
        # --- TIME-BASED STRATEGY LOGIC ---
        # Initialize signal as 0 (Neutral)
        target_signal = 0
        
        # 1. Buy before March 2024
        if curr_date < pd.Timestamp(2024, 3, 1):
            target_signal = 1
            
        # 2. Sell until September 2024 (March 1, 2024 to August 31, 2024)
        elif pd.Timestamp(2024, 3, 1) <= curr_date < pd.Timestamp(2024, 9, 1):
            target_signal = -1
            
        # 3. Buy until Jan 2025 (Sept 1, 2024 to Dec 31, 2024)
        elif pd.Timestamp(2024, 9, 1) <= curr_date < pd.Timestamp(2025, 1, 1):
            target_signal = 1
            
        # 4. Sell until April 2025 (Jan 1, 2025 to March 31, 2025)
        elif pd.Timestamp(2025, 1, 1) <= curr_date < pd.Timestamp(2025, 4, 1):
            target_signal = -1
            
        # 5. Buy until September 2025 (April 1, 2025 to August 31, 2025)
        elif pd.Timestamp(2025, 4, 1) <= curr_date < pd.Timestamp(2025, 9, 1):
            target_signal = 1
            
        # 6. Sell for the rest of the time (September 1, 2025 onwards)
        else:
            target_signal = -1

        # Use 20% of current liquid balance as the budget
        current_budget = engine.balance * 0.20

        # Execute at next row's open to maintain bias-free testing
        engine.execute_trade(
            signal=target_signal,
            budget=current_budget,
            market_price=next_row['open'], 
            timestamp=next_row['timestamp'],
            gas_fee=0.0 
        )

    # 6. Close any remaining open position
    if engine.current_position is not None:
        last_row = df.iloc[-1]
        exit_sig = -1 if engine.current_position['type'] == 'long' else 1
        engine.execute_trade(exit_sig, 0, last_row['close'], last_row['timestamp'], 0.0)

    # --- PERFORMANCE METRICS CALCULATION ---
    history = engine.get_logs()
    if history.empty:
        print("No trades executed.")
        return

    # A. Reconstructing PnL Value since engine.py only provides PnL%
    # pnl_val = (final_balance_of_this_trade - final_balance_of_previous_trade)
    # We use .diff() to find the change in balance per trade
    history['balance_prev'] = history['final_balance'].shift(1).fillna(initial_funds)
    history['pnl_val'] = history['final_balance'] - history['balance_prev']

    # B. Basic Stats
    net_profit = engine.balance - initial_funds
    total_trades = len(history)
    wins = history[history['pnl%'] > 0]
    losses = history[history['pnl%'] <= 0]
    win_rate = (len(wins) / total_trades) * 100

    # C. USDT/RS Averages
    avg_win = wins['pnl_val'].mean() if not wins.empty else 0
    avg_loss = losses['pnl_val'].mean() if not losses.empty else 0

    # D. Time Metrics (Duration & CAGR)
    history['entry_datetime'] = pd.to_datetime(history['entry_datetime'])
    history['exit_datetime'] = pd.to_datetime(history['exit_datetime'])
    avg_duration = (history['exit_datetime'] - history['entry_datetime']).mean()

    # Timeframe calculation
    days_total = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years_elapsed = max(days_total / 365.25, 0.001)
    cagr = (((engine.balance / initial_funds) ** (1 / years_elapsed)) - 1) * 100

    # E. Drawdown Calculation (Compounded)
    balance_curve = pd.Series([initial_funds] + history['final_balance'].tolist())
    peaks = balance_curve.cummax()
    drawdowns = (balance_curve - peaks) / peaks
    max_dd = drawdowns.min() * 100

    # F. Risk Ratios (Sharpe/Sortino/Calmar)
    trade_returns = history['pnl%'] / 100
    ann_factor = np.sqrt(total_trades / years_elapsed)
    
    sharpe = (trade_returns.mean() / trade_returns.std() * ann_factor) if trade_returns.std() != 0 else 0
    
    neg_ret = trade_returns[trade_returns < 0]
    sortino = (trade_returns.mean() / neg_ret.std() * ann_factor) if not neg_ret.empty and neg_ret.std() != 0 else 0
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # G. Benchmark (ETH & Mock SOL)
    eth_bh = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    # SOL Benchmark note: To get real SOL data, you'd need a separate CSV.
    # This prints the current ETH data as the reference.
    
    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print(f"{'FINAL PERFORMANCE REPORT':^50}")
    print("="*50)
    print(f"Net Profit:                {net_profit:,.2f} USDT")
    print(f"Net PnL :                  {net_profit/100000:.4f}")
    print(f"Total Closed Trades:       {total_trades}")
    print(f"Win Rate:                  {win_rate:.2f}%")
    print(f"Max Drawdown:              {max_dd:.2f}%")
    print(f"Avg Winning Trade:         {avg_win:,.2f} USDT")
    print(f"Avg Losing Trade:          {avg_loss:,.2f} USDT")
    print(f"Avg Holding Duration:      {avg_duration}")
    print("-" * 50)
    print(f"CAGR:                      {cagr:.2f}%")
    print(f"Sharpe Ratio:              {sharpe:.2f}")
    print(f"Sortino Ratio:             {sortino:.2f}")
    print(f"Calmar Ratio:              {calmar:.2f}")
    print("-" * 50)
    print(f"Buy and Hold (ETH):        {eth_bh:.2f}%")
    print("="*50)


    # 1. Prepare the data for plotting
    # We start with the initial balance at the start of the backtest
    times = [df['timestamp'].iloc[0]]
    balances = [initial_funds]

    # 2. Add the balance after every closed trade
    if not history.empty:
        times.extend(pd.to_datetime(history['exit_datetime']).tolist())
        balances.extend(history['final_balance'].tolist())

    # 3. Create the Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, balances, label='Portfolio Value (Equity)', color='#007bff', linewidth=2)
    
    # 4. Formatting the chart
    plt.title('Ethereum Algorithmic Trading: Portfolio Value Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Balance (USDT/RS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Fill the area under the curve for better visuals
    plt.fill_between(times, balances, initial_funds, where=(pd.Series(balances) >= initial_funds), 
                     interpolate=True, color='green', alpha=0.1)
    plt.fill_between(times, balances, initial_funds, where=(pd.Series(balances) < initial_funds), 
                     interpolate=True, color='red', alpha=0.1)

    # 5. Save and Show
    plt.tight_layout()
    plt.savefig('equity_curve.png')
    print("Graph saved as 'equity_curve.png'")
    plt.plot(df['timestamp'], (df['close'] / df['close'].iloc[0]) * initial_funds, label='Buy & Hold ETH', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    DATA_FILE = 'ETH-USDT_1h.csv' 
    run_backtest(DATA_FILE)