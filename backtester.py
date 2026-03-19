import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums & Constants ─────────────────────────────────────────────────────────

class TradeType(Enum):
    LONG  = "long"
    SHORT = "short"

class TradeStatus(Enum):
    OPEN   = "open"
    CLOSED = "closed"

TOTAL_COST_RATE  = 0.001   


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Represents a single round-trip trade."""
    trade_id        : int
    trade_type      : TradeType
    entry_datetime  : datetime
    entry_price     : float          # effective price AFTER slippage
    volume          : float          # units of ETH traded

    exit_datetime   : Optional[datetime] = None
    exit_price      : Optional[float]    = None   # effective price AFTER slippage
    pnl_pct         : Optional[float]    = None
    pnl_usd         : Optional[float]    = None
    status          : TradeStatus        = TradeStatus.OPEN

    # cost breakdown (stored for analysis)
    entry_cost_usd  : float = 0.0
    exit_cost_usd   : float = 0.0

    def summary(self) -> dict:
        return {
            "trade_id"       : self.trade_id,
            "trade_type"     : self.trade_type.value,
            "entry_datetime" : self.entry_datetime,
            "entry_price"    : round(self.entry_price, 4),
            "exit_datetime"  : self.exit_datetime,
            "exit_price"     : round(self.exit_price, 4) if self.exit_price else None,
            "volume"         : self.volume,
            "pnl_pct"        : round(self.pnl_pct, 4) if self.pnl_pct is not None else None,
            "pnl_usd"        : round(self.pnl_usd, 4) if self.pnl_usd is not None else None,
            "status"         : self.status.value,
        }


@dataclass
class Portfolio:
    """Tracks balance, open position, and trade history."""
    initial_balance : float
    balance         : float = field(init=False)
    open_trade      : Optional[Trade] = field(default=None, init=False)
    trade_history   : list[Trade]     = field(default_factory=list, init=False)
    _trade_counter  : int             = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.balance = self.initial_balance

    # ── helpers ───────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._trade_counter += 1
        return self._trade_counter

    @property
    def position(self) -> Optional[TradeType]:
        return self.open_trade.trade_type if self.open_trade else None

    def get_trade_log(self) -> pd.DataFrame:
        """Return all closed trades as a tidy DataFrame."""
        return pd.DataFrame([t.summary() for t in self.trade_history])

    def stats(self) -> dict:
        closed = [t for t in self.trade_history if t.status == TradeStatus.CLOSED]
        if not closed:
            return {"message": "No closed trades yet."}
        pnls = [t.pnl_usd for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "total_trades"    : len(closed),
            "win_rate"        : round(len(wins) / len(closed) * 100, 2),
            "total_pnl_usd"   : round(sum(pnls), 4),
            "avg_pnl_usd"     : round(np.mean(pnls), 4),
            "avg_win_usd"     : round(np.mean(wins), 4)   if wins   else 0,
            "avg_loss_usd"    : round(np.mean(losses), 4) if losses else 0,
            "final_balance"   : round(self.balance, 4),
            "return_pct"      : round((self.balance - self.initial_balance)
                                       / self.initial_balance * 100, 4),
        }


# ── Core Function ─────────────────────────────────────────────────────────────

def execute_trade(
    portfolio     : Portfolio,
    signal        : int,           #  1 → enter long / exit short
                                   # -1 → enter short / exit long
    timestamp     : datetime,      # current bar's timestamp (pass explicitly)
    market_price  : float,         # close (or wherever you want to fill)
    volume        : float,         # ETH units to trade
    verbose       : bool = True,
) -> Optional[Trade]:
    """
    Execute a trade or close an existing position based on `signal`.

    Signal logic
    ─────────────
     signal =  1  →  if SHORT open  → close it (buy-to-cover)
                      if no position → open LONG
     signal = -1  →  if LONG open   → close it (sell)
                      if no position → open SHORT

    Costs applied
    ─────────────
    Effective fill = market_price × (1 ± TOTAL_COST_RATE)
      LONG  entry  : price * (1 + rate)   ← you pay more
      LONG  exit   : price * (1 - rate)   ← you receive less
      SHORT entry  : price * (1 - rate)   ← you sell at less
      SHORT exit   : price * (1 + rate)   ← you buy back at more

    Returns the Trade object that was acted on (opened or closed).
    """
    if signal not in (1, -1):
        raise ValueError("signal must be 1 or -1")
    if volume <= 0:
        raise ValueError("volume must be positive")
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    current_position = portfolio.position
    trade_acted_on: Optional[Trade] = None

    # ── CLOSE existing position ───────────────────────────────────────────────
    if (current_position == TradeType.LONG  and signal == -1) or \
       (current_position == TradeType.SHORT and signal ==  1):

        t = portfolio.open_trade

        # effective exit price
        if t.trade_type == TradeType.LONG:
            effective_exit = market_price * (1 - TOTAL_COST_RATE)
        else:  # SHORT
            effective_exit = market_price * (1 + TOTAL_COST_RATE)

        exit_cost = market_price * t.volume * TOTAL_COST_RATE

        # PnL calculation
        if t.trade_type == TradeType.LONG:
            pnl_usd = (effective_exit - t.entry_price) * t.volume
        else:  # SHORT: profit when price falls
            pnl_usd = (t.entry_price - effective_exit) * t.volume

        pnl_pct = pnl_usd / (t.entry_price * t.volume) * 100

        # update trade record
        t.exit_datetime = timestamp
        t.exit_price    = effective_exit
        t.exit_cost_usd = exit_cost
        t.pnl_usd       = pnl_usd
        t.pnl_pct       = pnl_pct
        t.status        = TradeStatus.CLOSED

        # update portfolio
        portfolio.balance += pnl_usd
        portfolio.open_trade = None
        portfolio.trade_history.append(t)
        trade_acted_on = t

        if verbose:
            print(f"[{timestamp}] CLOSED {t.trade_type.value.upper():5s} | "
                  f"entry={t.entry_price:.2f}  exit={effective_exit:.2f}  "
                  f"vol={t.volume}  PnL={pnl_usd:+.2f} USD ({pnl_pct:+.3f}%)  "
                  f"balance={portfolio.balance:.2f}")

    # ── OPEN new position ─────────────────────────────────────────────────────
    #    (only if there's no conflicting open trade; after close, slot is free)
    if portfolio.open_trade is None:
        trade_type = TradeType.LONG if signal == 1 else TradeType.SHORT

        # effective entry price
        if trade_type == TradeType.LONG:
            effective_entry = market_price * (1 + TOTAL_COST_RATE)
        else:
            effective_entry = market_price * (1 - TOTAL_COST_RATE)

        entry_cost = market_price * volume * TOTAL_COST_RATE
        notional   = effective_entry * volume

        if notional > portfolio.balance:
            if verbose:
                print(f"[{timestamp}] Insufficient balance "
                      f"({portfolio.balance:.2f}) for notional {notional:.2f}. "
                      f"Trade skipped.")
            return trade_acted_on   # return whatever was closed (or None)

        # reserve capital
        portfolio.balance -= notional

        new_trade = Trade(
            trade_id       = portfolio._next_id(),
            trade_type     = trade_type,
            entry_datetime = timestamp,
            entry_price    = effective_entry,
            volume         = volume,
            entry_cost_usd = entry_cost,
        )
        portfolio.open_trade = new_trade
        trade_acted_on = new_trade

        if verbose:
            print(f"[{timestamp}] OPENED {trade_type.value.upper():5s} | "
                  f"entry={effective_entry:.2f}  vol={volume}  "
                  f"notional={notional:.2f}  balance={portfolio.balance:.2f}")

    return trade_acted_on