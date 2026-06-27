#!/usr/bin/env python3
"""
repro_bertopic.py  --  LimTopicのクラスタリング(③)を忠実再現

目的: 我々のBERTopic設定が known-good な出力(≈35トピックの正気な限界テーマ)を
出すかを、LimTopicの公開コーパスで検算する。Limsight本体エンジンの土台確認。

このサンドボックスはHuggingFaceに繋がらないため、ローカルで実行すること:
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas
    python repro_bertopic.py --csv "LimTopic/Datasets/All Data/df_all_data.csv"

埋め込み・パラメータはLimTopic準拠:
    embedding   : all-MiniLM-L6-v2   (彼らのcoherence最良)
    UMAP        : n_neighbors=13, n_components=7
    HDBSCAN     : min_cluster_size=10
    zeroshot sim: 0.75               (--guided 時のseed誘導しきい値)
"""

import argparse

import pandas as pd

from limtopic_clean import clean_corpus

# LimTopic Table 7 の seed words (テーマ追跡の初期語彙にも使える)
SEED_TOPICS = [
    "language and multilinguality", "dataset and corpus size",
    "computational cost and hardware", "machine translation",
    "tokenization and segmentation", "interpretability",
    "morphology and semantics", "memory", "skewed or biased distributions",
    "generalizability", "bias", "hyperparameter tuning",
    "real world robustness", "noisy data", "time and efficiency",
    "annotations", "evaluation and metrics", "diversity",
]


def build_model(guided: bool):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=13, n_components=7, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    kwargs = dict(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words="english"),
        min_topic_size=10,
        calculate_probabilities=False,
        verbose=True,
    )
    if guided:
        kwargs["zeroshot_topic_list"] = SEED_TOPICS
        kwargs["zeroshot_min_similarity"] = 0.75
    return BERTopic(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="LimTopicのdf_all_data.csv")
    ap.add_argument("--guided", action="store_true",
                    help="Table 7 seed wordsでzeroshot誘導する")
    ap.add_argument("--out", default="topics_assigned.csv")
    args = ap.parse_args()

    docs = clean_corpus(pd.read_csv(args.csv))
    print(f"clean docs: {len(docs)}")

    model = build_model(args.guided)
    topics, _ = model.fit_transform(docs)

    info = model.get_topic_info()
    n_topics = (info["Topic"] != -1).sum()
    noise = int(info.loc[info["Topic"] == -1, "Count"].sum()) if -1 in info["Topic"].values else 0
    print(f"\ntopics: {n_topics} | noise: {noise} ({100*noise/len(docs):.0f}%)\n")

    for _, row in info[info["Topic"] != -1].head(15).iterrows():
        t = row["Topic"]
        words = ", ".join(w for w, _ in model.get_topic(t)[:8])
        print(f"[topic {t:>2}] n={row['Count']:>3} | {words}")

    pd.DataFrame({"doc": docs, "topic": topics}).to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
