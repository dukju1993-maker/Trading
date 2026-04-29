"""변동성 돌파 전략(Larry Williams) 백테스트.

종목 5개(카카오, 네이버, 셀트리온, 에코프로비엠, KODEX 레버리지)에 대해
K = 0.3 / 0.5 / 0.7 세 가지 값으로 백테스트한다.

매수 기준가 = 당일 시가 + (전일 고가 - 전일 저가) * K
당일 고가가 매수 기준가를 돌파하면 그 가격에 매수, 당일 종가에 매도.

데이터 소스: pykrx (KRX 공식 데이터). 네트워크가 차단된 환경에서는
    --demo-data 플래그로 결정론적 합성 OHLCV를 사용해 로직을 검증할 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

TICKERS: dict[str, str] = {
    "035720": "Kakao",
    "035420": "Naver",
    "068270": "Celltrion",
    "247540": "EcoProBM",
    "122630": "KODEX_Leverage",
}

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
K_VALUES = (0.3, 0.5, 0.7)
INITIAL_CAPITAL = 10_000_000
BUY_COST = 0.00015      # 0.015%
SELL_COST = 0.00195     # 0.195% (수수료 + 세금)
SLIPPAGE = 0.001        # 0.1%
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def fetch_pykrx(ticker: str, start: str, end: str) -> pd.DataFrame:
    from pykrx import stock

    s = start.replace("-", "")
    e = end.replace("-", "")
    df = stock.get_market_ohlcv_by_date(s, e, ticker)
    if df is None or df.empty:
        raise RuntimeError(f"pykrx returned no data for {ticker}")
    df = df.rename(
        columns={"시가": "open", "고가": "high", "저가": "low",
                 "종가": "close", "거래량": "volume"}
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def synthesize(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Deterministic OHLCV for offline verification (NOT real prices)."""
    rng = np.random.default_rng(int(ticker))
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    drift = {"035720": 0.0001, "035420": 0.0002, "068270": -0.0001,
             "247540": 0.0003, "122630": 0.0004}.get(ticker, 0.0)
    vol = {"035720": 0.025, "035420": 0.020, "068270": 0.022,
           "247540": 0.035, "122630": 0.030}.get(ticker, 0.025)
    base = {"035720": 80000, "035420": 380000, "068270": 200000,
            "247540": 350000, "122630": 20000}.get(ticker, 50000)
    log_ret = rng.normal(drift, vol, size=n)
    close = base * np.exp(np.cumsum(log_ret))
    intraday_range = np.abs(rng.normal(0, vol, size=n)) + vol * 0.4
    high = close * (1 + intraday_range / 2)
    low = close * (1 - intraday_range / 2)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, vol / 4, size=n - 1))
    open_ = np.clip(open_, low, high)
    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def load_prices(use_demo: bool) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        if use_demo:
            data[ticker] = synthesize(ticker, START_DATE, END_DATE)
            continue
        try:
            data[ticker] = fetch_pykrx(ticker, START_DATE, END_DATE)
            print(f"[pykrx] {ticker} {TICKERS[ticker]}: {len(data[ticker])} bars")
        except Exception as exc:
            print(f"[pykrx] {ticker} failed ({exc}); falling back to demo data",
                  file=sys.stderr)
            data[ticker] = synthesize(ticker, START_DATE, END_DATE)
    return data


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
def breakout_signals(df: pd.DataFrame, k: float) -> tuple[pd.Series, pd.Series]:
    """Return (target_price, triggered_mask) for the breakout strategy."""
    prev_range = (df["high"].shift(1) - df["low"].shift(1))
    target = df["open"] + prev_range * k
    triggered = (df["high"] >= target) & prev_range.notna()
    return target, triggered.fillna(False)


def run_breakout(df: pd.DataFrame, k: float, capital: float) -> dict:
    """Backtest one stock for one K value.

    매수 체결가는 매수 기준가(target). 슬리피지·수수료는 체결가에 가산하여
    실효 진입가/청산가를 계산한 뒤 일별 수익률로 변환한다.
    """
    target, triggered = breakout_signals(df, k)
    eff_entry = target * (1 + SLIPPAGE) * (1 + BUY_COST)
    eff_exit = df["close"] * (1 - SLIPPAGE) * (1 - SELL_COST)

    daily_ret = pd.Series(0.0, index=df.index)
    daily_ret.loc[triggered] = (eff_exit[triggered] / eff_entry[triggered]) - 1.0

    equity = capital * (1 + daily_ret).cumprod()

    n_trades = int(triggered.sum())
    wins = int((daily_ret[triggered] > 0).sum())
    win_rate = wins / n_trades if n_trades else 0.0
    total_return = equity.iloc[-1] / capital - 1
    mdd = (equity / equity.cummax() - 1).min()

    return {
        "equity": equity,
        "daily_return": daily_ret,
        "triggered": triggered,
        "total_return": float(total_return),
        "mdd": float(mdd),
        "win_rate": float(win_rate),
        "trades": n_trades,
    }


def run_buy_and_hold(df: pd.DataFrame, capital: float) -> pd.Series:
    """첫날 시가 매수 → 마지막 날 종가 매도 (수수료·슬리피지 반영)."""
    entry_price = df["open"].iloc[0] * (1 + SLIPPAGE) * (1 + BUY_COST)
    shares = capital / entry_price
    equity = shares * df["close"]
    final_value = shares * df["close"].iloc[-1] * (1 - SLIPPAGE) * (1 - SELL_COST)
    equity = equity.copy()
    equity.iloc[-1] = final_value
    return equity


def vbt_stats_from_returns(returns: pd.Series, capital: float) -> dict:
    """Run returns through vectorbt for cross-checked summary stats."""
    synth_price = capital * (1 + returns).cumprod()
    pf = vbt.Portfolio.from_holding(close=synth_price, init_cash=capital, freq="1D")
    return {
        "vbt_total_return": float(pf.total_return()),
        "vbt_max_drawdown": float(pf.max_drawdown()),
        "vbt_sharpe": float(pf.sharpe_ratio()),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-data", action="store_true",
                        help="pykrx 대신 합성 데이터로 실행 (오프라인 검증용)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_prices(use_demo=args.demo_data)

    n = len(data)
    cap_per_stock = INITIAL_CAPITAL / n

    per_rows: list[dict] = []
    portfolios: dict[float, pd.Series] = {}

    for k in K_VALUES:
        portfolio_eq: pd.Series | None = None
        for ticker, df in data.items():
            res = run_breakout(df, k, cap_per_stock)
            vbt_extra = vbt_stats_from_returns(res["daily_return"], cap_per_stock)
            per_rows.append({
                "ticker": ticker,
                "name": TICKERS[ticker],
                "K": k,
                "total_return_%": round(res["total_return"] * 100, 2),
                "mdd_%": round(res["mdd"] * 100, 2),
                "win_rate_%": round(res["win_rate"] * 100, 2),
                "trades": res["trades"],
                "vbt_sharpe": round(vbt_extra["vbt_sharpe"], 3),
            })
            eq = res["equity"]
            portfolio_eq = eq if portfolio_eq is None else portfolio_eq.add(eq, fill_value=0)
        portfolios[k] = portfolio_eq

    # Buy & hold portfolio
    bh_equity: pd.Series | None = None
    for ticker, df in data.items():
        eq = run_buy_and_hold(df, cap_per_stock)
        bh_equity = eq if bh_equity is None else bh_equity.add(eq, fill_value=0)

    # Per-stock × K table
    df_per = pd.DataFrame(per_rows).sort_values(["K", "ticker"]).reset_index(drop=True)

    # Portfolio summary
    summary_rows: list[dict] = []
    for k, eq in portfolios.items():
        summary_rows.append({
            "strategy": f"VolatilityBreakout K={k}",
            "final_value": round(float(eq.iloc[-1]), 0),
            "total_return_%": round(eq.iloc[-1] / INITIAL_CAPITAL * 100 - 100, 2),
            "mdd_%": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        })
    summary_rows.append({
        "strategy": "Buy&Hold (equal weight)",
        "final_value": round(float(bh_equity.iloc[-1]), 0),
        "total_return_%": round(bh_equity.iloc[-1] / INITIAL_CAPITAL * 100 - 100, 2),
        "mdd_%": round(float((bh_equity / bh_equity.cummax() - 1).min()) * 100, 2),
    })
    df_summary = pd.DataFrame(summary_rows)

    # Save outputs
    per_csv = RESULTS_DIR / "per_stock_results.csv"
    sum_csv = RESULTS_DIR / "portfolio_summary.csv"
    df_per.to_csv(per_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(sum_csv, index=False, encoding="utf-8-sig")

    # Equity curve chart
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, eq in portfolios.items():
        ax.plot(eq.index, eq.values, label=f"VolBreakout K={k}", linewidth=1.5)
    ax.plot(bh_equity.index, bh_equity.values, "--", color="black",
            label="Buy & Hold", linewidth=1.5)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Volatility Breakout vs Buy & Hold — Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (KRW)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "equity_curves.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    # Drawdown chart
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for k, eq in portfolios.items():
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, label=f"K={k}")
    bh_dd = bh_equity / bh_equity.cummax() - 1
    ax2.plot(bh_dd.index, bh_dd.values * 100, "--", color="black", label="Buy & Hold")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    dd_path = RESULTS_DIR / "drawdown.png"
    fig2.savefig(dd_path, dpi=120)
    plt.close(fig2)

    # Console output
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print("\n=== 종목 × K 값별 결과 ===")
    print(df_per.to_string(index=False))
    print("\n=== 포트폴리오 요약 (초기 자본 1천만원, 1/N) ===")
    print(df_summary.to_string(index=False))
    print(f"\n저장: {per_csv}, {sum_csv}, {chart_path}, {dd_path}")


if __name__ == "__main__":
    main()
