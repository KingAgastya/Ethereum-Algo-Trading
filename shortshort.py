import pandas as pd
import numpy as np
import os
from engine import EthereumTradingEngine
import matplotlib.pyplot as plt

def calculate_ema(df, window=20):
    return df['close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2  
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(df, window=20, std_dev=2):
    sma = df['close'].rolling(window=window).mean()
    rstd = df['close'].rolling(window=window).std()
    upper_band = sma + (rstd * std_dev)
    lower_band = sma - (rstd * std_dev)
    return upper_band, sma, lower_band

def calculate_adx(df, window=14):
    """Average Directional Index (ADX) to measure trend strength"""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(df, window) 
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/window).mean() / tr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    return adx

def run_backtest(csv_path, initial_funds=50000.0):
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

    # --- INTEGRATED 2 YEAR LOGIC ---
    #df = df.head(17520).copy()
    df = df.tail(8855).copy()
    # -------------------------------

    # 4. Initialize Engine
    engine = EthereumTradingEngine(initial_balance=initial_funds, fee_rate=0.001)

    # 5. The "No-Bias" Loop
    # --- STEP 1: PRE-CALCULATE ALL INDICATORS ---
    df['ema_20'] = calculate_ema(df, window=20)
    df['ema_50'] = calculate_ema(df, window=50)
    df['ema_100'] = calculate_ema(df, window=100)
    df['macd'], df['macd_signal'], _ = calculate_macd(df)
    df['adx'] = calculate_adx(df, window=14)
    df['rsi'] = calculate_rsi(df, window=14)
    df['bb_up'], df['bb_mid'], df['bb_low'] = calculate_bollinger_bands(df, window=20, std_dev=2)
    v_mean = df['volume'].rolling(window=20).mean()
    v_std = df['volume'].rolling(window=20).std()
    df['volume_z'] = (df['volume'] - v_mean) / v_std
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].max(axis=1) - df['low']
    df['atr'] = calculate_atr(df, window=14)
    
    # --- TRACK WEALTH FOR SHARPE ---
    portfolio_values = []
    trend_trades = 0
    sideways_trades = 0
    prev_target_signal = 0  # For signal confirmation filter (improvement #3)

    for i in range(len(df) - 1):
        if i < 100: 
            portfolio_values.append(engine.balance)
            continue
        
        prev_row = df.iloc[i - 1]
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if pd.isna(current_row['adx']):
            portfolio_values.append(engine.balance)
            continue

        raw_signal = 0
        is_sideways_signal = False

        # --- TREND-STRENGTH BASED STRATEGY SWITCH (ADX) ---

        # 1. STRONG TREND (ADX > 30): MACD + TRIPLE EMA
        if current_row['adx'] > 30:
            if (current_row['macd'] > current_row['macd_signal']) and \
               (current_row['close'] > current_row['ema_20'] > current_row['ema_50'] > current_row['ema_100']):
                raw_signal = 1
                trend_trades += 1
            elif (current_row['macd'] < current_row['macd_signal']) and \
                 (current_row['close'] < current_row['ema_20'] < current_row['ema_50'] < current_row['ema_100']):
                raw_signal = -1
                trend_trades += 1
            else:
                if engine.current_position:
                    raw_signal = 1 if engine.current_position['type'] == 'long' else -1
                else:
                    raw_signal = 0

        # 2. WEAK TREND / RANGING (ADX < 25): RSI + BOLLINGER — LONG-BIASED
        elif current_row['adx'] < 25:
            if (current_row['rsi'] < 40) and (current_row['close'] < current_row['bb_low']):
                raw_signal = 1
                is_sideways_signal = True
                sideways_trades += 1
            elif (current_row['rsi'] >= 75) and \
                 (current_row['close'] > current_row['bb_up']) and \
                 (current_row['close'] < current_row['ema_50']):
                raw_signal = -1
                is_sideways_signal = True
                sideways_trades += 1
            else:
                if engine.current_position:
                    raw_signal = 1 if engine.current_position['type'] == 'long' else -1
                else:
                    raw_signal = 0

        # 3. TRANSITION ZONE (ADX 25-30): Hold current position, stay flat if none.
        #    ADX is noisy and oscillates constantly through this band — aggressively
        #    flattening here causes excessive exits, re-entry fees, and missed moves.
        else:
            if engine.current_position:
                raw_signal = 1 if engine.current_position['type'] == 'long' else -1
            else:
                raw_signal = 0

        # --- SIGNAL CONFIRMATION FILTER ---
        # Only applied to NEW trend direction entries (flat->new or long->short flip).
        # Sideways RSI+BB signals are reversal spikes — price bounces within one candle
        # so requiring confirmation would suppress them entirely. They fire immediately.
        # Hold signals (same direction as open position) also skip confirmation.
        if engine.current_position is not None:
            current_pos_signal = 1 if engine.current_position['type'] == 'long' else -1
            is_hold = (raw_signal == current_pos_signal)
        else:
            current_pos_signal = 0
            is_hold = False

        is_direction_change = (raw_signal != 0) and (raw_signal != current_pos_signal)

        if is_hold:
            target_signal = raw_signal
        elif is_sideways_signal:
            # Reversal spikes: act immediately, no confirmation
            target_signal = raw_signal
        elif is_direction_change and raw_signal == prev_target_signal:
            # Trend direction change confirmed on two consecutive candles
            target_signal = raw_signal
        elif is_direction_change:
            # First candle of a direction change — wait for confirmation
            target_signal = current_pos_signal if engine.current_position else 0
        else:
            target_signal = raw_signal

        prev_target_signal = raw_signal

        # ALL-IN 100% BUDGET
        # When a position is open, engine.balance is 0 (collateral is locked).
        # Use budget_allocated so that hold signals re-use the original capital,
        # and execute_trade closes + reopens correctly without a budget=0 bug.
        if engine.current_position is not None:
            current_budget = engine.current_position['budget_allocated']
        else:
            current_budget = engine.balance

        # Execute signal (long, short, or stay flat)
        if target_signal != 0:
            engine.execute_trade(
                signal=target_signal,
                budget=current_budget,
                market_price=next_row['open'],
                timestamp=next_row['timestamp'],
                gas_fee=0.0
            )
        else:
            # target_signal=0 means go flat: close position if open, else do nothing.
            # Use _close_position directly — NOT execute_trade(budget=0) — to avoid
            # the ghost-position / NaN pnl% bug.
            if engine.current_position is not None:
                engine._close_position(
                    market_price=next_row['open'],
                    timestamp=next_row['timestamp'],
                    gas_fee=0.0
                )

        # Calculate Hourly Portfolio Value
        current_val = engine.balance
        if engine.current_position is not None:
            pos = engine.current_position
            if pos['type'] == 'long':
                unrealized_pnl = (next_row['open'] - pos['entry_price']) * pos['volume']
                current_val += (pos['budget_allocated'] + unrealized_pnl)
            else:
                # For short: unrealized PnL = (entry - current) * volume
                unrealized_pnl = (pos['entry_price'] - next_row['open']) * pos['volume']
                current_val += (pos['budget_allocated'] + unrealized_pnl)
        portfolio_values.append(current_val)

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

    # A. Reconstructing PnL Value
    history['balance_prev'] = history['final_balance'].shift(1).fillna(initial_funds)
    history['pnl_val'] = history['final_balance'] - history['balance_prev']

    # --- RATIO CALCULATIONS ---
    pv_series = pd.Series(portfolio_values)
    hourly_returns = pv_series.pct_change().dropna()
    candles_per_year = 24 * 365 
    
    mean_ret = hourly_returns.mean()
    std_ret = hourly_returns.std()
    
    sharpe = (mean_ret * np.sqrt(candles_per_year)) / std_ret if std_ret != 0 else 0
    
    downside_returns = hourly_returns[hourly_returns < 0]
    std_down = downside_returns.std()
    sortino = (mean_ret * np.sqrt(candles_per_year)) / std_down if std_down != 0 else 0

    # B. Basic Stats
    net_profit = engine.balance - initial_funds
    total_trades = len(history)
    wins = history[history['pnl%'] > 0]
    losses = history[history['pnl%'] <= 0]
    win_rate = (len(wins) / total_trades) * 100

    # C. USDT/RS Averages
    avg_win = wins['pnl_val'].mean() if not wins.empty else 0
    avg_loss = losses['pnl_val'].mean() if not losses.empty else 0

    # D. Time Metrics
    history['entry_datetime'] = pd.to_datetime(history['entry_datetime'])
    history['exit_datetime'] = pd.to_datetime(history['exit_datetime'])
    avg_duration = (history['exit_datetime'] - history['entry_datetime']).mean()

    days_total = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years_elapsed = max(days_total / 365.25, 0.001)
    cagr = (((engine.balance / initial_funds) ** (1 / years_elapsed)) - 1) * 100

    # E. Drawdown Calculation
    balance_curve = pd.Series([initial_funds] + history['final_balance'].tolist())
    peaks = balance_curve.cummax()
    drawdowns = (balance_curve - peaks) / peaks
    max_dd = drawdowns.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    eth_bh = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    
    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print(f"{'FINAL PERFORMANCE REPORT':^50}")
    print("="*50)
    print(f"Net Profit:                {net_profit:,.2f} USDT")
    print(f"Net PnL Pct:               {100*net_profit/initial_funds:.4f} %")
    print(f"Total Closed Trades:       {total_trades}")
    print(f"Win Rate:                  {win_rate:.2f}%")
    print(f"Max Drawdown:              {max_dd:.2f}%")
    print(f"Trend Trades (High ADX):   {trend_trades}")
    print(f"Sideways Trades (Low ADX): {sideways_trades}")
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

    # 1. Prepare Plot Data
    times = [df['timestamp'].iloc[0]]
    balances = [initial_funds]
    if not history.empty:
        times.extend(pd.to_datetime(history['exit_datetime']).tolist())
        balances.extend(history['final_balance'].tolist())

    # 2. Create Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, balances, label='Portfolio Value (Equity)', color='#007bff', linewidth=2)
    plt.plot(df['timestamp'], (df['close'] / df['close'].iloc[0]) * initial_funds, label='Buy & Hold ETH', alpha=0.5, linestyle='--')
    
    plt.title('Ethereum Algorithmic Trading: Portfolio Value Over Time (2 Year Backtest)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Balance (USDT/RS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.fill_between(times, balances, initial_funds, where=(pd.Series(balances) >= initial_funds), 
                     interpolate=True, color='green', alpha=0.1)
    plt.fill_between(times, balances, initial_funds, where=(pd.Series(balances) < initial_funds), 
                     interpolate=True, color='red', alpha=0.1)

    plt.tight_layout()
    plt.savefig('equity_curve.png')
    plt.show()

if __name__ == "__main__":
    DATA_FILE = 'ETH-USDT_1h.csv' 
    run_backtest(DATA_FILE)