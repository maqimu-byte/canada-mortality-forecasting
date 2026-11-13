# Forecasting Canadian Mortality: Lee-Carter vs. Machine Learning

> 🚧 **Work in progress.** I'm building this project during my Master of AI (target: spring 2027). The roadmap below shows what is done and what is next.

## Why this matters

Life insurance pricing, reserving and retirement planning all rest on one question: **how fast will mortality keep improving?** Actuaries have long used the **Lee-Carter** model for this. This project asks whether machine-learning models can forecast age- and sex-specific death rates more accurately, and what they give up in interpretability and stability.

## Data

**Statistics Canada, Table 13-10-0837-01:** single-year complete life tables, Canada and provinces, by sex and single year of age, **1980–2024**. The main input is `qx`, the probability that a person aged *x* dies before *x + 1*.
Source: <https://doi.org/10.25318/1310083701-eng> (Statistics Canada Open Licence)

## Plan

- [ ] **Data pipeline:** download Table 13-10-0837-01 and reshape it into age × year matrices of `qx` / log death rates, by sex
- [ ] **EDA:** mortality surfaces (heatmaps), improvement rates by age band, the COVID-19 years (2020–2022) as a structural shock
- [ ] **Baseline:** Lee-Carter (SVD fit, random-walk-with-drift forecast of κₜ)
- [ ] **ML models:** gradient boosting on (age, year, sex) features, and a small neural network (e.g. an LSTM over κₜ, or a direct age-period model)
- [ ] **Backtesting:** train on 1980–2014, forecast 2015–2024; compare RMSE/MAPE on log mortality and on period life expectancy
- [ ] **Insurance view:** turn the forecasts into the expected present value of a 20-year term life benefit and show how much model choice moves the price
- [ ] **Write-up:** results, limitations (short series, pandemic years), next steps

## Planned structure

```
data/        raw downloads (git-ignored) + README with source info
notebooks/   01_eda · 02_lee_carter · 03_ml_models · 04_backtest · 05_term_pricing
src/         reusable code: data loading, Lee-Carter, evaluation
```

## Tech

Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib

---
*Qianmeng (Laura) Ma, Master of AI student (Seneca Polytechnic), CLU®.*
