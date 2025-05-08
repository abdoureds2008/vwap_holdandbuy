#classical vwap + hold and buy strateg
# multi_vwap_LS_adaptive_cost.py
"""
Adaptive VWAP mean-reversion baseline *with the same costs as the RL env*:
0.10 % commission + 0.10 % slippage per fill, 0.02 % daily borrow fee.
Universe: AAPL, MSFT, SBUX  •  2020-01-02 → 2020-07-30
"""

# ───────── Imports ─────────
import sys, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib, yfinance as yf
warnings.filterwarnings("ignore", category=FutureWarning)

# ───────── Parameters ──────
TICKERS   = ["AAPL","MSFT","SBUX"]
START,END = "2020-01-01","2020-07-31"         # END exclusive
INIT_CAP  = 20_000
ROLL_RET  = 21
ATR_WIN   = 14
GAP_K     = 0.25
EMA_FAST, EMA_SLOW = 20, 50

COST_PCT      = 0.001      # 0.10 % commission  (entry & exit)
SLIPPAGE_PCT  = 0.001      # 0.10 % slippage    (entry & exit)
BORROW_FEE    = 0.0002     # 0.02 % per short leg per day

FILES = dict(report="vwap_report.png", pnl="pnl_bars.png",
             curve="portfolio_vs_bh.png", results="vwap_backtest_results.csv",
             trades="trade_log.csv")

# ───────── 1  Prices ───────
print("Fetching",", ".join(TICKERS))
raw = yf.download(TICKERS,START,END,interval="1d",
                  auto_adjust=False,group_by="ticker",progress=False)
if raw.empty: sys.exit("No data.")
blocks=[]
for t in TICKERS:
    df = raw[t].copy()
    if isinstance(df.columns,pd.MultiIndex):
        df = df.xs("Price",axis=1,level=0) if "Price" in df.columns.get_level_values(0) else df.droplevel(1,axis=1)
    df.columns=[c.lower() for c in df.columns]
    df["ticker"],df["date"]=t,df.index
    blocks.append(df.reset_index(drop=True))
px=pd.concat(blocks)

# ───────── 2  Signals ──────
hlc=(px["high"]+px["low"]+px["close"])/3
px["vwap"]=hlc.groupby(px["ticker"]).transform(lambda s:s.rolling(5).mean())

tr=pd.concat([(px["high"]-px["low"]),
              (px["high"]-px.groupby("ticker")["close"].shift()).abs(),
              (px["low"] -px.groupby("ticker")["close"].shift()).abs()],axis=1).max(axis=1)
atr=tr.groupby(px["ticker"]).transform(lambda s:s.rolling(ATR_WIN).mean())
px["gap_thr"]=GAP_K*atr/px["close"]
px["delta"]=px["close"]/px["vwap"]-1

basket=(px.pivot(index="date",columns="ticker",values="close").mean(axis=1))
trend_up=(basket.ewm(span=EMA_FAST,adjust=False).mean() >
          basket.ewm(span=EMA_SLOW,adjust=False).mean()).astype(int)

long_cond  = px["delta"]<=-px["gap_thr"]
short_cond = px["delta"]>= px["gap_thr"]
raw_sig=np.select([long_cond,short_cond],[1,-1],0)
px["signal"]=np.where((raw_sig==-1)& trend_up.reindex(px["date"]).values,0,
                      np.where((raw_sig==1)&(~trend_up.reindex(px["date"]).values),0,raw_sig))
px["position"]=px.groupby("ticker")["signal"].shift(1).fillna(0)
px["ret"]=px.groupby("ticker")["close"].pct_change().fillna(0)

# ───────── 3  Weights & returns (pre-cost) ───────
vol=(px.pivot(index="date",columns="ticker",values="ret")
       .rolling(ROLL_RET).std())
inv_vol=1/vol
pos=(px.pivot(index="date",columns="ticker",values="position")
       .reindex(columns=TICKERS).fillna(0))
raw_w=pos*inv_vol
gross=raw_w.abs().sum(axis=1).replace(0,np.nan)
weights=raw_w.div(gross,axis=0).fillna(0)

rets=(px.pivot(index="date",columns="ticker",values="ret")
        .reindex(columns=TICKERS).fillna(0))

gross_turn = (weights.diff().abs().sum(axis=1)).fillna(0)   # one-way turnover
trade_cost = gross_turn * (COST_PCT+SLIPPAGE_PCT)           # paid twice (in/out) but diff() captures both legs

short_notional = (weights.shift().clip(upper=0).abs()
                  * px.pivot(index="date",columns="ticker",values="close")
                  ).sum(axis=1).fillna(0)
borrow_cost = short_notional / INIT_CAP * BORROW_FEE        # % of initial capital

strat_ret_raw = (weights.shift().fillna(0)*rets).sum(axis=1)
strat_ret = strat_ret_raw - trade_cost - borrow_cost

bh_ret = rets.mean(axis=1)

# ───────── 4  Equity & risk ───────
eq_ls,eq_bh = (1+strat_ret).cumprod(),(1+bh_ret).cumprod()
pv_ls,pv_bh = eq_ls*INIT_CAP,eq_bh*INIT_CAP
pnl         = pv_ls.diff().fillna(0)
dd_ls,dd_bh = eq_ls/eq_ls.cummax()-1 , eq_bh/eq_bh.cummax()-1
rollS=(strat_ret.rolling(ROLL_RET).mean()/strat_ret.rolling(ROLL_RET).std())*np.sqrt(252)
rollV=strat_ret.rolling(ROLL_RET).std()*np.sqrt(252)

# ───────── 5  Trade-log (unchanged helper) ──────
def snap_prev(idx,ts): return ts if ts in idx else idx[max(idx.searchsorted(ts,'right')-1,0)]
rows=[]
for t in TICKERS:
    df=px[px["ticker"]==t].set_index("date")
    flips=df["position"].diff().fillna(0)
    if df["position"].iloc[0]!=0: flips.iloc[0]=df["position"].iloc[0]
    turns=flips[flips!=0].index.to_list()
    for i,en_raw in enumerate(turns):
        ex_raw=turns[i+1]-pd.Timedelta(days=1) if i+1<len(turns) else df.index[-1]
        en=snap_prev(df.index,en_raw); ex=snap_prev(df.index,ex_raw)
        side="LONG" if df.at[en,"position"]>0 else "SHORT"
        r=(df.at[ex,"close"]/df.at[en,"close"]-1)*(1 if side=="LONG" else -1)
        rows.append([t,side,en,ex,(ex-en).days+1,r])
pd.DataFrame(rows,columns=["ticker","side","entry","exit","days","return"])\
  .to_csv(FILES["trades"],index=False)

# ───────── 6  Summary print ───────
def perf(c):
    tot=c.iloc[-1]-1
    cagr=c.iloc[-1]**(252/len(c))-1
    dd=(c/c.cummax()-1).min()
    vol=c.pct_change().std()*np.sqrt(252)
    shr=(c.pct_change().mean()*252)/vol
    return tot,cagr,dd,vol,shr

summary=pd.DataFrame({"Buy&Hold":perf(eq_bh),"VWAP_cost":perf(eq_ls)},
                     index=["Tot","CAGR","MaxDD","Vol","Sharpe"])\
        .applymap(lambda x:f"{x:.2%}" if abs(x)<10 else f"{x:.2f}")
print(summary,"\n")

# ───────── 7  Graphics (unchanged) ──────
plt.figure(figsize=(14,16))
plt.subplot(3,2,1);plt.plot(pv_bh,label="Buy&Hold");plt.plot(pv_ls,label="Adaptive VWAP (cost)");plt.legend();plt.title(f"Portfolio value (start ${INIT_CAP:,})")
plt.subplot(3,2,2);plt.plot(eq_bh,label="Buy&Hold");plt.plot(eq_ls,label="Adaptive VWAP (cost)");plt.legend();plt.title("Equity curve")
plt.subplot(3,2,3);plt.plot(dd_bh,label="Buy&Hold");plt.plot(dd_ls,label="Adaptive VWAP (cost)");plt.legend();plt.title("Drawdown")
plt.subplot(3,2,4);plt.bar(pnl.index,pnl);plt.title("Daily P/L (net of costs)");plt.ylabel("US-$")
plt.subplot(3,2,5);plt.plot(rollS);plt.title(f"{ROLL_RET}-day rolling Sharpe")
plt.subplot(3,2,6);plt.plot(rollV);plt.title(f"{ROLL_RET}-day rolling vol")
plt.tight_layout();plt.savefig(FILES["report"],dpi=150)

# extras unchanged …

# ───────── 8  Save CSV ──────
pd.DataFrame({"eq_bh":eq_bh,"eq_ls":eq_ls,"pv_bh":pv_bh,"pv_ls":pv_ls,
              "dd_bh":dd_bh,"dd_ls":dd_ls,"rollS":rollS,"rollV":rollV,
              "strat_ret":strat_ret,"bh_ret":bh_ret}).to_csv(FILES["results"])

print("Outputs saved → PNGs • CSV • trade log")
