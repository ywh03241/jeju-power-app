# -*- coding: utf-8 -*-
"""
학습 스크립트: jeju_2025_merged.csv로 모형을 학습해 model.pkl 저장
실행: python train.py   (데이터를 바꾸면 다시 실행)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from model import X_main, X_mlp, controls

DATA = "jeju_2025_merged.csv"


def cp_X(d, Tc, Th):
    X = pd.DataFrame({"CDD": np.maximum(d["T"] - Tc, 0), "HDD": np.maximum(Th - d["T"], 0)})
    return pd.concat([X.reset_index(drop=True), controls(d["시"], d["휴일"])], axis=1)


def find_base_temps(d, grid=np.arange(10, 26.5, 0.5)):
    best = (-np.inf, None, None)
    for Tc in grid:
        for Th in grid[grid <= Tc]:
            X = cp_X(d, Tc, Th)
            r2 = LinearRegression().fit(X, d["y"]).score(X, d["y"])
            if r2 > best[0]:
                best = (r2, Tc, Th)
    return float(best[1]), float(best[2])


def fit_models(d, Tc, Th):
    main = LinearRegression().fit(X_main(d["T"], d["H"], d["시"], d["휴일"], Tc, Th), d["y"])
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=42),
    ).fit(X_mlp(d["T"], d["H"], d["시"], d["휴일"]), d["y"])
    return main, mlp


def scores(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAPE(%)": mean_absolute_percentage_error(y, p) * 100}


def main():
    df = pd.read_csv(DATA, parse_dates=["일시", "날짜"])
    df = df.rename(columns={"기온_평균": "T", "습도_평균": "H", "전력수요_MW": "y"})
    df["T"] = df["T"].astype(float)
    df["H"] = df["H"].astype(float)

    # 1) 성능 확인용: 매달 마지막 7일 검증
    va_mask = df["날짜"].dt.day > df["날짜"].dt.days_in_month - 7
    tr, va = df[~va_mask].reset_index(drop=True), df[va_mask].reset_index(drop=True)
    Tc, Th = find_base_temps(tr)
    m_main, m_mlp = fit_models(tr, Tc, Th)
    s_main = scores(va["y"], m_main.predict(X_main(va["T"], va["H"], va["시"], va["휴일"], Tc, Th)))
    s_mlp = scores(va["y"], m_mlp.predict(X_mlp(va["T"], va["H"], va["시"], va["휴일"])))

    # 2) 배포용: 1년 전체로 다시 학습 (기준온도는 1)에서 찾은 값 사용)
    full = df.reset_index(drop=True)
    m_main, m_mlp = fit_models(full, Tc, Th)

    # 3) 하루 기온·습도 패턴(월별) - 최저/최고기온만 넣어도 24시간 곡선을 만들기 위함
    g = df.groupby("날짜")
    df["Tmin"], df["Tmax"], df["Hmean"] = g["T"].transform("min"), g["T"].transform("max"), g["H"].transform("mean")
    df["shape"] = ((df["T"] - df["Tmin"]) / (df["Tmax"] - df["Tmin"]).replace(0, np.nan)).fillna(0.5)
    df["hdev"] = df["H"] - df["Hmean"]
    t_shape = {m: df[df["월"] == m].groupby("시")["shape"].mean().reindex(range(1, 25)).values for m in range(1, 13)}
    h_dev = {m: df[df["월"] == m].groupby("시")["hdev"].mean().reindex(range(1, 25)).values for m in range(1, 13)}
    month_typical = df.groupby("월").agg(tmin=("Tmin", "mean"), tmax=("Tmax", "mean"), hmean=("Hmean", "mean")).round(1)

    # 4) 피크 경고 기준 (2025년 시간별 수요 상위 10%, 3%)
    warn, alert = float(df["y"].quantile(0.90)), float(df["y"].quantile(0.97))

    daily = df.groupby("날짜").agg(T=("T", "mean"), y=("y", "mean"), 계절=("계절", "first")).reset_index()

    bundle = {
        "main": m_main, "mlp": m_mlp, "Tc": Tc, "Th": Th,
        "scores": {"③ 기준온도+습도": s_main, "④ MLP": s_mlp},
        "t_shape": t_shape, "h_dev": h_dev, "month_typical": month_typical,
        "warn": warn, "alert": alert, "daily": daily,
        "max_2025": float(df["y"].max()),
    }
    joblib.dump(bundle, "model.pkl")

    c = dict(zip(X_main([0], [0], [1], [0], Tc, Th).columns, m_main.coef_))
    print(f"기준온도: 냉방 Tc={Tc}℃, 난방 Th={Th}℃")
    print(f"냉방 기울기 {c['CDD']:.1f} MW/℃, 난방 기울기 {c['HDD']:.1f} MW/℃")
    print(f"검증 MAPE: ③ {s_main['MAPE(%)']:.2f}%, ④ {s_mlp['MAPE(%)']:.2f}%")
    print(f"피크 경고 기준: 주의 {warn:.0f} MW, 경고 {alert:.0f} MW")
    print("저장: model.pkl")


if __name__ == "__main__":
    main()
