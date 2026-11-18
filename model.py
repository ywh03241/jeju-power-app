# -*- coding: utf-8 -*-
"""
공통 모듈: 입력변수 만들기, 예측, 수요 분해
train.py(학습)와 app.py(화면)가 같이 사용한다.
"""
import numpy as np
import pandas as pd

H_REF = 75  # 습도 기준값(%). 습도 효과는 이 값과의 차이로 계산


def controls(hours, offday):
    """통제변수: 시간대 더미(1시 기준, 2~24시) + 휴일"""
    hours = np.asarray(hours)
    X = pd.DataFrame({f"h_{h}": (hours == h).astype(float) for h in range(2, 25)})
    X["휴일"] = np.asarray(offday, dtype=float)
    return X


def X_main(T, H, hours, offday, Tc, Th):
    """③ 기준온도 분리 + 온습도 상호작용 모형의 입력변수"""
    T, H = np.asarray(T, float), np.asarray(H, float)
    cdd = np.maximum(T - Tc, 0)
    X = pd.DataFrame({
        "CDD": cdd,
        "HDD": np.maximum(Th - T, 0),
        "H": H,
        "CDDxH": cdd * (H - H_REF),
    })
    return pd.concat([X, controls(hours, offday)], axis=1)


def X_mlp(T, H, hours, offday):
    """④ MLP 입력변수"""
    hours = np.asarray(hours)
    return pd.DataFrame({
        "T": np.asarray(T, float),
        "H": np.asarray(H, float),
        "휴일": np.asarray(offday, float),
        "hsin": np.sin(2 * np.pi * hours / 24),
        "hcos": np.cos(2 * np.pi * hours / 24),
    })


def predict(bundle, T, H, hours, offday):
    """주 모형(③) 예측 + 수요 분해(기본/냉방/난방/습도)"""
    m, Tc, Th = bundle["main"], bundle["Tc"], bundle["Th"]
    X = X_main(T, H, hours, offday, Tc, Th)
    c = dict(zip(X.columns, m.coef_))
    pred = m.predict(X)

    cool = c["CDD"] * X["CDD"]
    heat = c["HDD"] * X["HDD"]
    hum = c["H"] * (X["H"] - H_REF) + c["CDDxH"] * X["CDDxH"]
    base = pred - cool - heat - hum  # 절편 + 시간대 + 휴일 + 습도 기준값 몫
    return pd.DataFrame({
        "시": np.asarray(hours),
        "기온": np.asarray(T, float),
        "습도": np.asarray(H, float),
        "예측수요": pred,
        "기본수요": base,
        "냉방": cool.values,
        "난방": heat.values,
        "습도효과": hum.values,
    })


def predict_mlp(bundle, T, H, hours, offday):
    return bundle["mlp"].predict(X_mlp(T, H, hours, offday))


def make_hourly_weather(bundle, month, tmin, tmax, hmean):
    """최저·최고기온, 평균습도만으로 24시간 기온·습도 곡선 만들기 (2025년 같은 달의 평균 하루 패턴 이용)"""
    shape = bundle["t_shape"][month]   # 0(그날 최저)~1(그날 최고) 사이 값, 1~24시
    hdev = bundle["h_dev"][month]      # 그날 평균습도 대비 시간별 차이
    T = tmin + shape * (tmax - tmin)
    H = np.clip(hmean + hdev, 5, 100)
    return np.round(T, 1), np.round(H, 1)
