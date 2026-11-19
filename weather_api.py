# -*- coding: utf-8 -*-
"""
Open-Meteo 예보 불러오기
- 학습 자료와 같은 4개 ASOS 관측소 위치의 시간별 기온·습도 예보를 받아 평균
- 무료, 인증키 없음 (비상업적 용도). 데이터 라이선스 CC BY 4.0 → 화면에 출처 표시
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"
STATIONS = {  # 기상청 ASOS 관측소 위치(위도, 경도)
    "제주": (33.5141, 126.5297),
    "고산": (33.2938, 126.1628),
    "서귀포": (33.2462, 126.5653),
    "성산": (33.3868, 126.8802),
}
MAX_AHEAD_DAYS = 14   # 오늘부터 며칠 뒤까지 예측 가능
MAX_PAST_DAYS = 7     # 며칠 전까지 다시 볼 수 있음


def fetch_forecast(day: dt.date, timeout=15):
    """
    day의 1~24시 기온·습도(4개 관측소 평균)를 돌려준다.
    전력 자료 규칙에 맞춰 'h시' = 그날 0시 + h시간 (24시 = 다음 날 0시).
    반환: (DataFrame[시, 기온, 습도], 관측소별 DataFrame)
    """
    names = list(STATIONS)
    params = {
        "latitude": ",".join(str(STATIONS[n][0]) for n in names),
        "longitude": ",".join(str(STATIONS[n][1]) for n in names),
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Asia/Seoul",
        "start_date": day.isoformat(),
        "end_date": (day + dt.timedelta(days=1)).isoformat(),
    }
    r = requests.get(URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):  # 위치가 1개면 dict, 여러 개면 list로 온다
        data = [data]

    want = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in range(1, 25)]
    per = {}
    for name, loc in zip(names, data):
        h = loc["hourly"]
        s = pd.DataFrame({"T": h["temperature_2m"], "H": h["relative_humidity_2m"]},
                         index=pd.to_datetime(h["time"]))
        per[name] = s.reindex(want)

    T = pd.concat({n: d["T"] for n, d in per.items()}, axis=1)
    H = pd.concat({n: d["H"] for n, d in per.items()}, axis=1)
    out = pd.DataFrame({
        "시": np.arange(1, 25),
        "기온": T.mean(axis=1).round(1).values,
        "습도": H.mean(axis=1).round(1).values,
    })
    if out[["기온", "습도"]].isna().any().any():
        raise ValueError("예보 자료에 빈 시간이 있습니다.")
    stations = pd.DataFrame({"시": np.arange(1, 25)})
    for n in names:
        stations[f"기온_{n}"] = T[n].values
    return out, stations
