import pandas as pd
import numpy as np
import os
from engine_2 import EthereumTradingEngine
import matplotlib.pyplot as plt

# =========================
# INDICATORS
# =========================

def calculate_ema(df, window):
    return df['close'].ewm(span=window, adjust=False).mean()


def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, window=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(df, window=20, std_dev=2):
    sma = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    return sma + std_dev * std, sma, sma - std_dev * std


def calculate_adx(df, window=14):
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)

    tr = calculate_atr(df, window)

    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/window).mean()

# =========================
# POSITION SIZING
# =========================

BASE_ALLOC = {
    'trend_long': 1.00,
    'trend_short': 0.70,
    'side_long': 0.80,
    'side_short': 0.50,
}


def get_position_size(capital, signal_type, atr, atr_mean):
    base = BASE_ALLOC[signal_type]

    if pd.notna(atr) and pd.notna(atr_mean) and atr_mean > 0:
        scalar = np.clip(atr_mean / atr, 0.80, 1.00)
    else:
        scalar = 1.0

    return capital * base * scalar

# =========================
# BACKTEST
# =========================

def run_backtest(csv_path, initial_funds=50000.0):

    if os.path.exists('logs.csv'):
        os.remove('logs.csv')

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    engine = EthereumTradingEngine(initial_balance=initial_funds, fee_rate=0.001)
    
    #df = df.head(17520).copy()
    #df = df.tail(8855).copy()

    # Indicators
    df['ema_20'] = calculate_ema(df, 20)
    df['ema_50'] = calculate_ema(df, 50)
    df['ema_100'] = calculate_ema(df, 100)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['adx'] = calculate_adx(df)
    df['rsi'] = calculate_rsi(df)
    df['bb_up'], df['bb_mid'], df['bb_low'] = calculate_bollinger_bands(df)
    df['atr'] = calculate_atr(df)
    df['atr_mean'] = df['atr'].rolling(20).mean()

    portfolio_values = []

    for i in range(len(df) - 1):
        if i < 100:
            portfolio_values.append(engine.balance)
            continue

        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if pd.isna(row['adx']):
            portfolio_values.append(engine.balance)
            continue

        signal = 0
        signal_type = None

        # Trend
        if row['adx'] > 30:
            if row['macd'] > row['macd_signal'] and row['close'] > row['ema_20'] > row['ema_50'] > row['ema_100']:
                signal, signal_type = 1, 'trend_long'
            elif row['macd'] < row['macd_signal'] and row['close'] < row['ema_20'] < row['ema_50'] < row['ema_100']:
                signal, signal_type = -1, 'trend_short'

        # Range
        elif row['adx'] < 25:
            if row['rsi'] < 40 and row['close'] < row['bb_low']:
                signal, signal_type = 1, 'side_long'
            elif row['rsi'] > 65 and row['close'] > row['bb_up']:
                signal, signal_type = -1, 'side_short'

        # Hold logic
        if engine.current_position is not None and signal == 0:
            signal = 1 if engine.current_position['type'] == 'long' else -1

        # Position sizing
        if engine.current_position:
            budget = engine.current_position['budget_allocated']
        elif signal_type:
            budget = get_position_size(engine.balance, signal_type, row['atr'], row['atr_mean'])
        else:
            budget = engine.balance

        # Execute
        if signal == 1:
            engine.execute_trade(signal, budget, next_row['open'], next_row['timestamp'], 0.0)
        elif signal == -1 and engine.current_position and engine.current_position['type'] == 'long':
            engine._close_position(next_row['open'], next_row['timestamp'], 0)

        # Portfolio value
        val = engine.balance
        if engine.current_position:
            pos = engine.current_position
            pnl = (next_row['open'] - pos['entry_price']) * pos['volume']
            val += pos['budget_allocated'] + pnl

        portfolio_values.append(val)

    # Close final position
    if engine.current_position:
        last = df.iloc[-1]
        engine.execute_trade(-1, 0, last['close'], last['timestamp'], 0.0)

    # =========================
    # PERFORMANCE METRICS
    # =========================

    history = engine.get_logs()
    if history.empty:
        print("No trades executed.")
        return

    history['balance_prev'] = history['final_balance'].shift(1).fillna(initial_funds)
    history['pnl_val'] = history['final_balance'] - history['balance_prev']

    pv = pd.Series(portfolio_values)
    returns = pv.pct_change().dropna()

    candles_per_year = 24 * 365

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret * np.sqrt(candles_per_year)) / std_ret if std_ret != 0 else 0

    downside = returns[returns < 0]
    sortino = (mean_ret * np.sqrt(candles_per_year)) / downside.std() if downside.std() != 0 else 0

    net_profit = engine.balance - initial_funds
    total_trades = len(history)
    win_rate = (len(history[history['pnl%'] > 0]) / total_trades) * 100

    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years = max(days / 365.25, 0.001)
    cagr = ((engine.balance / initial_funds) ** (1 / years) - 1) * 100

    equity = pd.Series([initial_funds] + history['final_balance'].tolist())
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks
    max_dd = drawdown.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    bh = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

    print("\n" + "="*50)
    print(f"{'FINAL PERFORMANCE REPORT':^50}")
    print("="*50)
    print(f"Net Profit:      {net_profit:,.2f}")
    print(f"Return:          {100*net_profit/initial_funds:.2f}%")
    print(f"Trades:          {total_trades}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Max Drawdown:    {max_dd:.2f}%")
    print("-"*50)
    print(f"CAGR:            {cagr:.2f}%")
    print(f"Sharpe:          {sharpe:.2f}")
    print(f"Sortino:         {sortino:.2f}")
    print(f"Calmar:          {calmar:.2f}")
    print("-"*50)
    print(f"Buy & Hold:      {bh:.2f}%")
    print("="*50)

        # =========================
    # PLOTTING
    # =========================

    times = [df['timestamp'].iloc[0]]
    balances = [initial_funds]

    if not history.empty:
        times.extend(pd.to_datetime(history['exit_datetime']).tolist())
        balances.extend(history['final_balance'].tolist())

    plt.figure(figsize=(12, 6))

    plt.plot(times, balances, label='Portfolio Value (Equity)', linewidth=2)

    plt.plot(
        df['timestamp'],
        (df['close'] / df['close'].iloc[0]) * initial_funds,
        label='Buy & Hold ETH',
        linestyle='--',
        alpha=0.5
    )

    plt.title('Ethereum Algorithmic Trading: Portfolio Value Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Balance')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.fill_between(
        times, balances, initial_funds,
        where=(pd.Series(balances) >= initial_funds),
        interpolate=True,
        alpha=0.1
    )

    plt.fill_between(
        times, balances, initial_funds,
        where=(pd.Series(balances) < initial_funds),
        interpolate=True,
        alpha=0.1
    )

    plt.tight_layout()
    plt.savefig('equity_curve.png')
    plt.show()
    
        # =========================
    # DRAWDOWN CURVE (WITH BUY & HOLD)
    # =========================

    # Strategy equity
    equity_curve = pd.Series(balances, index=pd.to_datetime(times))
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    # Buy & Hold equity
    bh_equity = (df['close'] / df['close'].iloc[0]) * initial_funds
    bh_equity.index = pd.to_datetime(df['timestamp'])
    bh_running_max = bh_equity.cummax()
    bh_drawdown = (bh_equity - bh_running_max) / bh_running_max

    plt.figure(figsize=(12, 5))

    plt.plot(drawdown.index, drawdown.values, label='Strategy Drawdown', linewidth=2)
    plt.plot(bh_drawdown.index, bh_drawdown.values, linestyle='--', label='Buy & Hold Drawdown')

    plt.title('Drawdown Comparison: Strategy vs Buy & Hold', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        where=(drawdown.values < 0),
        interpolate=True,
        alpha=0.15
    )

    plt.tight_layout()
    plt.savefig('drawdown_comparison.png')
    plt.show()

        # =========================
    # BUY / SELL SIGNAL PLOT
    # =========================

    # Ensure datetime
    history['entry_datetime'] = pd.to_datetime(history['entry_datetime'])
    history['exit_datetime'] = pd.to_datetime(history['exit_datetime'])

    # Price series
    price = df.set_index('timestamp')['close']

    plt.figure(figsize=(14, 6))

    # Plot price
    plt.plot(price.index, price.values, label='ETH Price', linewidth=1.5)

    # Plot BUY signals (entries)
    plt.scatter(
        history['entry_datetime'],
        df.set_index('timestamp').loc[history['entry_datetime'], 'close'],
        marker='^',
        s=80,
        color = 'green',
        label='Buy Signal'
    )

    # Plot SELL signals (exits)
    plt.scatter(
        history['exit_datetime'],
        df.set_index('timestamp').loc[history['exit_datetime'], 'close'],
        marker='v',
        s=80,
        color = 'red',
        label='Sell Signal'
    )

    plt.title('Buy and Sell Signals on Price Curve', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('signals_plot.png')
    plt.show()

if __name__ == "__main__":
    run_backtest('ETH-USDT_1h.csv')