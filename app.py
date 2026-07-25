from pathlib import Path
import csv
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

DATA_FILE = Path(__file__).with_name("pc_builds.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_data():
    with DATA_FILE.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def recommend(query, budget, top_k=3):
    data = [r for r in load_data() if int(r["price_yen"]) <= budget]
    if not data:
        return []

    model = load_model()
    texts = [r["use_case"] for r in data]
    text_vecs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    query_vec = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_vec, text_vecs)[0]
    top = scores.topk(k=min(top_k, len(data)))

    results = []
    for score, index in zip(top.values, top.indices):
        row = dict(data[int(index)])
        row["similarity"] = float(score)
        row["remaining_budget"] = budget - int(row["price_yen"])
        results.append(row)
    return results

TEST_CASES = [
    ("Apexを高フレームレートで遊びたい", 220000, "競技FPS"),
    ("YouTube用の動画を編集しながらゲームもしたい", 300000, "ゲーム＋動画編集"),
    ("ローカル環境で画像生成AIを動かしたい", 350000, "生成AI"),
    ("重いゲームをWQHDの高画質で楽しみたい", 280000, "WQHDゲーム"),
    ("4K解像度で最新ゲームを遊びたい", 360000, "4Kゲーム"),
    ("OBSで配信しながらFPSをプレイしたい", 290000, "ゲーム配信"),
    ("3DCG作品のレンダリングを高速化したい", 390000, "3D制作"),
    ("大学のレポート作成と軽いゲームに使いたい", 160000, "学生向け"),
    ("大規模な街づくりゲームを快適に動かしたい", 270000, "シミュレーション"),
    ("VRヘッドセットでゲームを滑らかに遊びたい", 300000, "VRゲーム"),
]

def evaluate():
    rows = []
    top1_count = 0
    top3_count = 0

    for text, budget, expected in TEST_CASES:
        results = recommend(text, budget, 3)
        predicted = results[0]["category"] if results else "候補なし"
        similarity = results[0]["similarity"] if results else 0.0
        top3 = [r["category"] for r in results]

        top1_ok = predicted == expected
        top3_ok = expected in top3
        top1_count += int(top1_ok)
        top3_count += int(top3_ok)

        rows.append({
            "入力文": text,
            "予算": f"{budget:,}円",
            "期待カテゴリ": expected,
            "1位カテゴリ": predicted,
            "類似度": round(similarity, 3),
            "Top-1": "○" if top1_ok else "×",
            "Top-3": "○" if top3_ok else "×",
        })

    return pd.DataFrame(rows), top1_count, top3_count

st.set_page_config(page_title="PC構成推薦", page_icon="🖥️", layout="centered")
st.title("🖥️ 用途・予算別 PC構成推薦")

search_tab, eval_tab = st.tabs(["構成を検索", "性能評価"])

with search_tab:
    query = st.text_area(
        "どんな用途で使いますか？",
        "モンハンをWQHDで高画質かつ快適に遊びたい",
        height=90
    )
    budget_man = st.slider("予算（万円）", 14, 60, 28)
    top_k = st.selectbox("表示する候補数", [1, 3, 5], index=1)

    if st.button("おすすめ構成を検索", type="primary", use_container_width=True):
        results = recommend(query.strip(), budget_man * 10000, top_k)
        if not results:
            st.error("予算内の候補がありません。")
        else:
            for i, row in enumerate(results, 1):
                with st.container(border=True):
                    st.subheader(f"候補{i}：{row['category']}")
                    st.caption(f"類似度 {row['similarity']:.3f} ／ 登録用途：{row['use_case']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("CPU", row["cpu"])
                        st.metric("RAM", row["ram"])
                        st.metric("電源", row["psu"])
                    with c2:
                        st.metric("GPU", row["gpu"])
                        st.metric("SSD", row["ssd"])
                        st.metric("概算価格", f"{int(row['price_yen']):,}円")
                    st.write(f"**推薦理由：** {row['reason']}")
                    st.write(f"**予算残額：** {row['remaining_budget']:,}円")

with eval_tab:
    st.subheader("言い換え入力10件による一括評価")
    st.write("ボタンを1回押すだけで10件を評価します。")

    if st.button("10件を一括評価", type="primary", use_container_width=True):
        with st.spinner("評価中です…"):
            df, top1_count, top3_count = evaluate()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Top-1正解率", f"{top1_count / 10 * 100:.1f}%")
        with c2:
            st.metric("Top-3正解率", f"{top3_count / 10 * 100:.1f}%")

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write(f"Top-1：{top1_count}/10件、Top-3：{top3_count}/10件")
