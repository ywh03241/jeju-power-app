# -*- coding: utf-8 -*-
"""
제주 전력수요 예보기 (PT#2)
실행: streamlit run app.py
"""
import datetime as dt
import io

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model import predict, predict_mlp, make_hourly_weather

st.set_page_config(page_title="제주 전력수요 예보기", page_icon="⚡", layout="wide")
HOURS = np.arange(1, 25)


@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")


B = load_bundle()
Tc, Th = B["Tc"], B["Th"]

# ------------------------------------------------------------------
# 사이드바: 입력
# ------------------------------------------------------------------
st.sidebar.header("① 날짜")
day = st.sidebar.date_input("예측할 날짜", dt.date.today() + dt.timedelta(days=1))
offday = st.sidebar.checkbox("휴일 (주말·공휴일)", value=day.weekday() >= 5,
                             help="주말은 자동으로 체크됩니다. 공휴일이면 직접 체크하세요.")

st.sidebar.header("② 기상 예보")
mode = st.sidebar.radio("입력 방식", ["최저·최고기온으로 간단 입력", "CSV 업로드 (24시간)"])
typ = B["month_typical"].loc[day.month]

if mode.startswith("최저"):
    c1, c2 = st.sidebar.columns(2)
    tmin = c1.number_input("최저기온(℃)", -10.0, 40.0, float(typ["tmin"]), 0.5)
    tmax = c2.number_input("최고기온(℃)", -10.0, 45.0, float(typ["tmax"]), 0.5)
    hmean = st.sidebar.slider("평균습도(%)", 10, 100, int(round(typ["hmean"])))
    if tmax < tmin:
        st.sidebar.error("최고기온이 최저기온보다 낮습니다.")
        st.stop()
    T0, H0 = make_hourly_weather(B, day.month, tmin, tmax, hmean)
    st.sidebar.caption(f"기본값은 2025년 {day.month}월 평균입니다. 시간별 값은 아래 표에서 직접 고칠 수 있어요.")
    wx = pd.DataFrame({"시": HOURS, "기온": T0, "습도": H0})
    wx = st.sidebar.data_editor(wx, key=f"ed_{day.month}_{tmin}_{tmax}_{hmean}", hide_index=True,
                                disabled=["시"], height=300)
else:
    tmpl = pd.DataFrame({"시": HOURS, "기온": "", "습도": ""}).to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button("입력 양식 내려받기", tmpl, "forecast_template.csv", "text/csv")
    up = st.sidebar.file_uploader("시·기온·습도 열이 있는 CSV (24행)", type="csv")
    if up is None:
        st.info("왼쪽에서 CSV를 올리면 예측이 시작됩니다. 양식을 내려받아 기온·습도만 채우면 돼요.")
        st.stop()
    raw = up.getvalue()
    try:
        wx = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        wx = pd.read_csv(io.BytesIO(raw), encoding="cp949")
    if not {"시", "기온", "습도"} <= set(wx.columns) or len(wx) != 24:
        st.error("CSV에는 '시', '기온', '습도' 열과 24개 행(1~24시)이 있어야 합니다.")
        st.stop()
    wx = wx.sort_values("시")[["시", "기온", "습도"]]

wx = wx.astype(float)
if wx.isna().any().any():
    st.error("기온·습도에 빈칸이 있습니다.")
    st.stop()

T, H = wx["기온"].values, wx["습도"].values
off = np.full(24, int(offday))

res = predict(B, T, H, HOURS, off)
res["MLP"] = predict_mlp(B, T, H, HOURS, off)


def level(v):
    if v >= B["alert"]:
        return "🔴 경고", "2025년 시간별 수요 상위 3% 수준"
    if v >= B["warn"]:
        return "🟠 주의", "2025년 시간별 수요 상위 10% 수준"
    return "🟢 보통", "평상 수준"


# ------------------------------------------------------------------
# 본문
# ------------------------------------------------------------------
st.title("⚡ 제주 전력수요 예보기")
st.caption(f"기온·습도 예보로 제주 시간별 전력수요를 예측합니다 · 학습자료: 2025년 제주 시간별 전력수요, "
           f"ASOS 4개 관측소(제주·고산·서귀포·성산) 평균")

tab1, tab2, tab3 = st.tabs(["📈 예측", "🌡️ 시나리오", "📚 모형 설명"])

# ---------------- 탭1: 예측 ----------------
with tab1:
    pk = res.loc[res["예측수요"].idxmax()]
    lv, lv_desc = level(pk["예측수요"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("피크 수요", f"{pk['예측수요']:.0f} MW")
    m2.metric("피크 시각", f"{int(pk['시'])}시")
    m3.metric("하루 평균", f"{res['예측수요'].mean():.0f} MW")
    m4.metric("피크 등급", lv, help=lv_desc)
    m4.caption(lv_desc)

    long = res.melt(id_vars="시", value_vars=["예측수요", "MLP"], var_name="모형", value_name="MW")
    long["모형"] = long["모형"].map({"예측수요": "③ 기준온도+습도 (주 모형)", "MLP": "④ MLP (비교)"})
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("시:Q", scale=alt.Scale(domain=[1, 24]), axis=alt.Axis(tickCount=24, title="시")),
        y=alt.Y("MW:Q", scale=alt.Scale(zero=False), title="전력수요 (MW)"),
        color=alt.Color("모형:N", legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("모형:N", legend=None),
        tooltip=["시", "모형", alt.Tooltip("MW:Q", format=".0f")],
    )
    rules = pd.DataFrame({"MW": [B["warn"], B["alert"]], "기준": ["주의", "경고"]})
    rule = alt.Chart(rules).mark_rule(strokeDash=[4, 4], color="gray").encode(y="MW:Q")
    rule_txt = alt.Chart(rules).mark_text(align="left", dx=4, dy=-6, color="gray").encode(
        x=alt.value(0), y="MW:Q", text="기준:N")
    st.subheader("시간별 예측")
    st.altair_chart((lines + rule + rule_txt).properties(height=340), width="stretch")

    st.subheader("날씨 때문에 늘어난 수요 (주 모형 분해)")
    st.caption("예측수요 = 기본수요(시간대·휴일에 따른 생활 수요) + 아래 세 가지. 습도효과는 습도 75% 기준이라 건조하면 음수가 됩니다.")
    comp = res.melt(id_vars="시", value_vars=["냉방", "난방", "습도효과"], var_name="구성", value_name="MW")
    bars = alt.Chart(comp).mark_bar().encode(
        x=alt.X("시:O", title="시", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sum(MW):Q", title="날씨 몫 (MW)"),
        color=alt.Color("구성:N", scale=alt.Scale(
            domain=["냉방", "난방", "습도효과"], range=["#d1495b", "#3d6fb6", "#2a9d8f"]),
            legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["시", "구성", alt.Tooltip("MW:Q", format=".1f")],
    ).properties(height=280)
    st.altair_chart(bars, width="stretch")
    s = res[["냉방", "난방", "습도효과"]].sum()
    tot = res["예측수요"].sum()
    st.caption(f"하루 전체 수요 중 냉방 몫 {s['냉방'] / tot * 100:.1f}%, 난방 몫 {s['난방'] / tot * 100:.1f}%, "
               f"습도 몫 {s['습도효과'] / tot * 100:+.1f}%")

    out = res.round(1)
    st.download_button("예측 결과 CSV 내려받기", out.to_csv(index=False).encode("utf-8-sig"),
                       f"predict_{day}.csv", "text/csv")
    with st.expander("시간별 표 보기"):
        st.dataframe(out, hide_index=True, width="stretch")

# ---------------- 탭2: 시나리오 ----------------
with tab2:
    st.write("예보보다 더 덥거나 습하면 피크가 얼마나 오를까? 폭염·한파 대비용으로 기온·습도를 바꿔 봅니다.")
    c1, c2 = st.columns(2)
    dT = c1.slider("기온 변화 (℃)", -6.0, 6.0, 2.0, 0.5)
    dH = c2.slider("습도 변화 (%p)", -30, 30, 0, 5)
    sc = predict(B, T + dT, np.clip(H + dH, 5, 100), HOURS, off)
    pk0, pk1 = res["예측수요"].max(), sc["예측수요"].max()
    a, b, c = st.columns(3)
    a.metric("예보 그대로 피크", f"{pk0:.0f} MW")
    b.metric("시나리오 피크", f"{pk1:.0f} MW", f"{pk1 - pk0:+.0f} MW ({(pk1 / pk0 - 1) * 100:+.1f}%)",
             delta_color="inverse")
    c.metric("시나리오 피크 등급", level(pk1)[0])
    both = pd.concat([res[["시", "예측수요"]].assign(구분="예보 그대로"),
                      sc[["시", "예측수요"]].assign(구분=f"기온 {dT:+.1f}℃ · 습도 {dH:+d}%p")])
    ch = alt.Chart(both).mark_line(point=True).encode(
        x=alt.X("시:Q", scale=alt.Scale(domain=[1, 24]), axis=alt.Axis(tickCount=24)),
        y=alt.Y("예측수요:Q", scale=alt.Scale(zero=False), title="전력수요 (MW)"),
        color=alt.Color("구분:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["시", "구분", alt.Tooltip("예측수요:Q", format=".0f")],
    ).properties(height=340)
    st.altair_chart(ch + rule + rule_txt, width="stretch")
    if Th < T.mean() < Tc and abs(dT) <= 2:
        st.info(f"오늘 기온이 냉·난방이 거의 필요 없는 구간({Th}~{Tc}℃)이라 기온을 조금 바꿔도 수요가 크게 변하지 않습니다.")

# ---------------- 탭3: 모형 설명 ----------------
with tab3:
    st.subheader("제주의 기온-전력수요는 U자 모양")
    d = B["daily"]
    pts = alt.Chart(d).mark_circle(size=40, opacity=0.75).encode(
        x=alt.X("T:Q", title="일평균 기온 (℃)"),
        y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="일평균 전력수요 (MW)"),
        color=alt.Color("계절:N", scale=alt.Scale(domain=["봄", "여름", "가을", "겨울"],
                                                  range=["#4c9f70", "#d1495b", "#e08e45", "#3d6fb6"])),
        tooltip=[alt.Tooltip("날짜:T"), alt.Tooltip("T:Q", format=".1f"), alt.Tooltip("y:Q", format=".0f")],
    )
    vr = pd.DataFrame({"T": [Th, Tc], "lab": [f"난방 기준 {Th}℃", f"냉방 기준 {Tc}℃"]})
    vl = alt.Chart(vr).mark_rule(strokeDash=[5, 5], color="gray").encode(x="T:Q")
    vt = alt.Chart(vr).mark_text(dy=-150, dx=4, align="left", color="gray").encode(x="T:Q", text="lab:N")
    st.altair_chart((pts + vl + vt).properties(height=380), width="stretch")

    coef = dict(zip(["CDD", "HDD"], B["main"].coef_[:2]))
    st.markdown(f"""
**주 모형: ③ 기준온도 분리 + 온습도 상호작용**
- 기온이 **{Tc}℃보다 높으면 냉방**, **{Th}℃보다 낮으면 난방** 수요가 생긴다고 보고 기울기를 따로 둡니다.
- 냉방: {Tc}℃ 위로 1℃ 오를 때마다 약 **{coef['CDD']:.1f} MW** 증가
- 난방: {Th}℃ 아래로 1℃ 내릴 때마다 약 **{coef['HDD']:.1f} MW** 증가
- 습도는 냉방 구간에서 냉방 효과를 키우는 항(냉방도 × 습도)으로 들어갑니다.
- 시간대(1~24시)와 휴일은 통제변수로 함께 넣었습니다.

정확도는 MLP가 조금 더 높지만, 주 모형은 **왜 수요가 오르는지(냉방/난방/습도)를 나눠 보여줄 수 있어서** 예보기의 기본 모형으로 썼습니다.
""")
    sc_df = pd.DataFrame(B["scores"]).T.round(2)
    st.write("검증 성능 (2025년 매달 마지막 7일)")
    st.dataframe(sc_df, width="stretch")
    st.caption("한계: 2025년 1년 자료만 학습 · 해안 관측소 4곳 평균 · 관광객 수 등 기상 외 요인은 반영하지 않음")
