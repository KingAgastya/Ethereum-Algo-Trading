import pandas as pd
import csv
import os

class EthereumTradingEngine:
    def __init__(self, initial_balance=100000.0, fee_rate=0.001):
        """
        initial_balance: Starting capital (e.g., in RS or USDT)
        fee_rate: 0.001 represents 0.1% (slippage + commission)
        """
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.current_position = None
        self.trade_history = []

    def execute_trade(self, signal, budget, market_price, timestamp, gas_fee=1.0):
        """
        signal: 1 (Long/Close Short), -1 (Short/Close Long)
        budget: The total amount of money (RS) you want to allocate to this trade.
        """
        # 1. Close existing positions if signal is opposite
        if self.current_position is not None:
            pos_type = self.current_position['type']
            if (pos_type == 'long' and signal == -1) or (pos_type == 'short' and signal == 1):
                self._close_position(market_price, timestamp, gas_fee)

        # 2. Open new position if we are flat
        if self.current_position is None:
            if budget > self.balance:
                print(f"[{timestamp}] REJECTED: Budget {budget} exceeds Balance {self.balance:.2f}")
                return
            
            if signal == 1:
                self._open_position('long', budget, market_price, timestamp, gas_fee)
            elif signal == -1:
                self._open_position('short', budget, market_price, timestamp, gas_fee)

    def _open_position(self, trade_type, budget, market_price, timestamp, gas_fee=0.0):
        available_after_gas = budget - gas_fee

        if trade_type == 'long':
            # Buy ETH at market price + fee (slippage)
            execution_price = market_price * (1 + self.fee_rate)
            volume = available_after_gas / execution_price
            # Full budget leaves the balance — it is now held as ETH
            self.balance -= budget

        else:
            # Short: we sell ETH we don't own.
            # The full budget is posted as collateral and locked out of balance.
            # We sell at market price - fee (we get slightly less).
            execution_price = market_price * (1 - self.fee_rate)
            volume = available_after_gas / execution_price
            # Deduct the full budget from balance (collateral locked)
            self.balance -= budget

        self.current_position = {
            'type': trade_type,
            'budget_allocated': budget,
            'volume': volume,
            'entry_price': execution_price,
            'entry_datetime': timestamp,
            'entry_gas': gas_fee
        }
        print(f"[{timestamp}] OPEN {trade_type.upper()} | Budget: {budget:.2f} | Net Vol: {volume:.4f} ETH @ {execution_price:.2f}")

    def _close_position(self, market_price, timestamp, gas_fee):
        pos = self.current_position

        if pos['type'] == 'long':
            # Sell ETH at market price - fee
            exit_price = market_price * (1 - self.fee_rate)
            exit_value = (pos['volume'] * exit_price) - gas_fee
            pnl_val = exit_value - (pos['budget_allocated'] - pos['entry_gas'])
            # Return whatever the ETH is now worth
            self.balance += exit_value

        else:
            # Short close: buy back ETH at market price + fee
            exit_price = market_price * (1 + self.fee_rate)
            # Cost to buy back the ETH we shorted
            buyback_cost = (pos['volume'] * exit_price) + gas_fee
            # PnL = what we sold for (entry_price * volume) minus what we buy back for
            # Entry commission was already baked into the lower execution_price at open
            pnl_val = (pos['entry_price'] * pos['volume']) - buyback_cost
            # Return collateral + profit (or collateral - loss)
            self.balance += pos['budget_allocated'] + pnl_val

        pnl_pct = (pnl_val / pos['budget_allocated']) * 100

        self.trade_history.append({
            'trade_type': pos['type'],
            'entry_datetime': pos['entry_datetime'],
            'entry_price': round(pos['entry_price'], 2),
            'exit_datetime': timestamp,
            'exit_price': round(exit_price, 2),
            'pnl%': round(pnl_pct, 4),
            'final_balance': round(self.balance, 2)
        })

        file_exists = os.path.isfile('logs.csv')
        f = open('logs.csv', 'a', newline='')
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "entry_datetime", "entry_price",
                "exit_datetime", "exit_price",
                "pnl%", "trade_type"
            ])
        w.writerow([pos['entry_datetime'], pos['entry_price'], timestamp, exit_price, pnl_pct, pos['type']])
        f.close()

        print(f"[{timestamp}] CLOSE {pos['type'].upper()} | Exit Price: {exit_price:.2f} | PnL%: {pnl_pct:.2f}%")
        self.current_position = None

    def get_logs(self):
        return pd.DataFrame(self.trade_history)