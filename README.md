# Adaptive VWAP Mean-Reversion Strategy (with Cost Simulation)

This project implements a **classical VWAP-based long-short mean-reversion strategy** with **adaptive trend filtering** and **realistic trading costs**, coded entirely in Python and backtested on daily data.

---

## 📈 Strategy Description

- **Approach**: VWAP (Volume Weighted Average Price) Mean-Reversion with trend confirmation.
- **Universe**: AAPL, MSFT, SBUX
- **Period**: 2020-01-02 to 2020-07-30
- **Capital**: $20,000 initial investment
- **Costs Included**:
  - 0.10% commission per trade
  - 0.10% slippage per trade
  - 0.02% daily borrow fee (short-selling cost)

---

## 🧠 Methodology Highlights

- **Signal Generation**:
  - Price deviation from 5-day VWAP
  - Trend confirmation using EMA(20) and EMA(50) on basket index
  - ATR-based gap threshold for entry signal

- **Portfolio Allocation**:
  - Inverse volatility weighting across selected stocks
  - Dynamic rebalancing with cost-aware trade execution

- **Performance Metrics**:
  - Equity curve, drawdown, rolling Sharpe & volatility
  - Trade log with individual PnLs
  - Buy & Hold benchmark comparison

---

## 🛠️ Dependencies

- Python 3.6+
- `numpy`, `pandas`, `matplotlib`
- `yfinance` (for fetching historical data)

Install dependencies:

```bash
pip install numpy pandas matplotlib yfinance
