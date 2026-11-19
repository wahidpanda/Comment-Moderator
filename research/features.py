"""Shared sklearn feature pipeline for LR / SVM / SGD."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


def make_tfidf_pipeline(
    *,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    max_features: int = 200_000,
    min_df: int = 2,
) -> FeatureUnion:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def attach_classifier(
    vectorizers: FeatureUnion,
    estimator,
) -> Pipeline:
    return Pipeline([("features", vectorizers), ("clf", estimator)])