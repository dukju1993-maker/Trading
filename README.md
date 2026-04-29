# 변동성 돌파 전략 백테스트

`pykrx` + `vectorbt` 기반의 Larry Williams 변동성 돌파 전략 백테스트.

## 개요

- **종목**: 카카오(035720), 네이버(035420), 셀트리온(068270), 에코프로비엠(247540), KODEX 레버리지(122630)
- **기간**: 2021-01-01 ~ 2025-12-31
- **전략**: 매수 기준가 = 당일 시가 + (전일 고가 − 전일 저가) × **K**
  당일 고가가 기준가를 돌파하면 기준가에 매수, 당일 종가에 청산.
- **K 값**: 0.3 / 0.5 / 0.7 비교
- **비용**: 매수 0.015%, 매도(수수료+세금) 0.195%, 슬리피지 0.1%
- **자본 배분**: 초기 1,000만원 → 종목당 1/N(=200만원) 균등

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 실제 KRX 데이터 사용 (인터넷 + KRX 접근 가능 환경)
python volatility_breakout.py

# 오프라인 검증용 합성 데이터 (실제 시세 아님)
python volatility_breakout.py --demo-data
```

> **주의**: `pykrx`가 KRX 서버에서 시세를 받아오므로, 방화벽이나 사내망에서
> `data.krx.co.kr`이 차단된 경우 데이터 호출이 실패할 수 있다. 이때는
> `--demo-data` 플래그로 결정론적 합성 OHLCV를 사용해 백테스트 로직만 검증할 수 있다.

## 산출물 (`results/`)

| 파일 | 설명 |
|------|------|
| `per_stock_results.csv` | 종목 × K값 별 수익률, MDD, 승률, 거래횟수, vbt Sharpe |
| `portfolio_summary.csv` | K별 포트폴리오 + 단순 매수후보유 비교 |
| `equity_curves.png` | K별 누적 수익곡선 vs Buy & Hold |
| `drawdown.png` | K별 낙폭 곡선 vs Buy & Hold |

## 구현 메모

- 매수 체결가 = `target × (1 + 슬리피지) × (1 + 매수수수료)` (실효 진입가)
- 매도 체결가 = `종가 × (1 − 슬리피지) × (1 − 매도수수료)` (실효 청산가)
- 일별 전략 수익률 = `eff_exit / eff_entry − 1` (돌파한 날만, 그 외 0)
- 종목별 자본을 독립적으로 굴린 뒤 일자별로 합산해 포트폴리오 가치 산출
- vectorbt는 일별 전략 리턴을 `Portfolio.from_holding`으로 흘려보내 Sharpe 등
  추가 통계를 교차 검증하는 용도로 사용
