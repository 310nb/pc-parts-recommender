from pathlib import Path
import csv

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util


DATA_FILE = Path(__file__).with_name("pc_builds.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_data
def load_data() -> list[dict[str, str]]:
    """推薦候補のPC構成をCSVから読み込む。"""
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {
        "use_case",
        "price_yen",
        "cpu",
        "gpu",
        "ram",
        "ssd",
        "psu",
        "category",
        "reason",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("pc_builds.csv の列構成が不正です。")
    return rows


@st.cache_resource(show_spinner="Sentence-BERTモデルを読み込んでいます…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="推薦候補をベクトル化しています…")
def load_corpus_embeddings():
    """全候補を一度だけベクトル化し、検索のたびの再計算を避ける。"""
    model = load_model()
    texts = [row["use_case"] for row in load_data()]
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def recommend(query: str, budget: int, top_k: int = 3) -> list[dict]:
    """予算内の候補を、入力文とのコサイン類似度順に返す。"""
    query = query.strip()
    if not query:
        return []
    if budget <= 0 or top_k <= 0:
        return []

    all_rows = load_data()
    eligible_indices = [
        index
        for index, row in enumerate(all_rows)
        if int(row["price_yen"]) <= budget
    ]
    if not eligible_indices:
        return []

    model = load_model()
    corpus_embeddings = load_corpus_embeddings()
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    eligible_embeddings = corpus_embeddings[eligible_indices]
    scores = util.cos_sim(query_embedding, eligible_embeddings)[0]
    top = scores.topk(k=min(top_k, len(eligible_indices)))

    results = []
    for score, local_index in zip(top.values, top.indices):
        row = dict(all_rows[eligible_indices[int(local_index)]])
        row["similarity"] = float(score)
        row["remaining_budget"] = budget - int(row["price_yen"])
        results.append(row)
    return results


TEST_CASES = [
    ("Apexを高フレームレートで遊びたい", 220_000, "競技FPS"),
    ("YouTube用の動画を編集しながらゲームもしたい", 300_000, "ゲーム＋動画編集"),
    ("ローカル環境で画像生成AIを動かしたい", 350_000, "生成AI"),
    ("重いゲームをWQHDの高画質で楽しみたい", 280_000, "WQHDゲーム"),
    ("4K解像度で最新ゲームを遊びたい", 360_000, "4Kゲーム"),
    ("OBSで配信しながらFPSをプレイしたい", 290_000, "ゲーム配信"),
    ("3DCG作品のレンダリングを高速化したい", 390_000, "3D制作"),
    ("大学のレポート作成と軽いゲームに使いたい", 160_000, "学生向け"),
    ("大規模な街づくりゲームを快適に動かしたい", 270_000, "シミュレーション"),
    ("VRヘッドセットでゲームを滑らかに遊びたい", 300_000, "VRゲーム"),
]


def evaluate() -> tuple[pd.DataFrame, int, int]:
    """固定した10件についてTop-1/Top-3カテゴリ正解率を測る。"""
    rows = []
    top1_count = 0
    top3_count = 0

    for text, budget, expected in TEST_CASES:
        results = recommend(text, budget, 3)
        predicted = results[0]["category"] if results else "候補なし"
        similarity = results[0]["similarity"] if results else 0.0
        top3 = [row["category"] for row in results]

        top1_ok = predicted == expected
        top3_ok = expected in top3
        top1_count += int(top1_ok)
        top3_count += int(top3_ok)

        rows.append(
            {
                "入力文": text,
                "予算": f"{budget:,}円",
                "期待カテゴリ": expected,
                "1位カテゴリ": predicted,
                "類似度": round(similarity, 3),
                "Top-1": "○" if top1_ok else "×",
                "Top-3": "○" if top3_ok else "×",
            }
        )

    return pd.DataFrame(rows), top1_count, top3_count


st.set_page_config(
    page_title="PC構成推薦",
    page_icon="🖥️",
    layout="centered",
)
st.title("🖥️ 用途・予算別 PC構成推薦")
st.caption("Sentence-BERTで入力文の意味を比較し、予算内の構成を推薦します。")

search_tab, eval_tab = st.tabs(["構成を検索", "性能評価"])

with search_tab:
    query = st.text_area(
        "どんな用途で使いますか？",
        "モンハンをWQHDの高画質かつ快適に遊びたい",
        height=90,
    )
    budget_man = st.slider("予算（万円）", 14, 60, 28)
    top_k = st.selectbox("表示する候補数", [1, 3, 5], index=1)

    if st.button("おすすめ構成を検索", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("用途を入力してください。")
        else:
            results = recommend(query, budget_man * 10_000, top_k)
            if not results:
                st.error("予算内の候補がありません。")
            else:
                for rank, row in enumerate(results, 1):
                    with st.container(border=True):
                        st.subheader(f"候補{rank}：{row['category']}")
                        st.caption(
                            f"類似度 {row['similarity']:.3f} ／ "
                            f"登録用途：{row['use_case']}"
                        )
                        left, right = st.columns(2)
                        with left:
                            st.metric("CPU", row["cpu"])
                            st.metric("RAM", row["ram"])
                            st.metric("電源", row["psu"])
                        with right:
                            st.metric("GPU", row["gpu"])
                            st.metric("SSD", row["ssd"])
                            st.metric("概算価格", f"{int(row['price_yen']):,}円")
                        st.write(f"**推薦理由：** {row['reason']}")
                        st.write(f"**予算残額：** {row['remaining_budget']:,}円")

with eval_tab:
    st.subheader("言い換え入力10件による一括評価")
    st.write(
        "登録文をそのまま使わない10件の入力で、"
        "期待カテゴリが上位候補に含まれるかを評価します。"
    )

    if st.button("10件を一括評価", type="primary", use_container_width=True):
        with st.spinner("評価中です…"):
            dataframe, top1_count, top3_count = evaluate()

        left, right = st.columns(2)
        with left:
            st.metric("Top-1正解率", f"{top1_count / len(TEST_CASES) * 100:.1f}%")
        with right:
            st.metric("Top-3正解率", f"{top3_count / len(TEST_CASES) * 100:.1f}%")

        st.dataframe(dataframe, use_container_width=True, hide_index=True)
        st.write(
            f"Top-1：{top1_count}/{len(TEST_CASES)}件、"
            f"Top-3：{top3_count}/{len(TEST_CASES)}件"
        )
