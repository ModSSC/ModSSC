from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

TEXT_CATALOG: dict[str, DatasetSpec] = {
    "webkb_course_cotraining": DatasetSpec(
        key="webkb_course_cotraining",
        provider="webkb1998",
        uri="webkb1998:course",
        modality="text",
        task="classification",
        description=(
            "WebKB Course subset used by Blum and Mitchell (1998): 1,051 paired "
            "full-page/inlink-anchor documents, 230 course and 821 non-course pages, "
            "with no official split."
        ),
        required_extra=None,
        source_kwargs={},
        homepage=("https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-51/www/co-training/data/"),
        license=None,
        citation=(
            "Blum, A., & Mitchell, T. (1998). Combining Labeled and Unlabeled Data "
            "with Co-Training. COLT, 92-100. https://doi.org/10.1145/279943.279962."
        ),
    ),
    "ag_news": DatasetSpec(
        key="ag_news",
        provider="hf",
        uri="hf:ag_news",
        modality="text",
        task="classification",
        description="AG News (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={"text_column": "text", "label_column": "label", "prefer_test_split": True},
    ),
    "imdb": DatasetSpec(
        key="imdb",
        provider="hf",
        uri="hf:imdb",
        modality="text",
        task="classification",
        description="IMDB (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={"text_column": "text", "label_column": "label", "prefer_test_split": True},
    ),
    "amazon_polarity": DatasetSpec(
        key="amazon_polarity",
        provider="hf",
        uri="hf:amazon_polarity",
        modality="text",
        task="classification",
        description="Amazon-2 sentiment (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={
            "text_column": "content",
            "label_column": "label",
            "prefer_test_split": True,
        },
    ),
    "amazon_reviews_multi_en": DatasetSpec(
        key="amazon_reviews_multi_en",
        provider="hf",
        uri="hf:amazon_reviews_multi/en",
        modality="text",
        task="classification",
        description="Amazon-5 reviews (amazon_reviews_multi/en). Labels are 1-5.",
        required_extra="hf",
        source_kwargs={
            "text_column": "review_body",
            "label_column": "stars",
            "prefer_test_split": True,
        },
    ),
    "dbpedia_14": DatasetSpec(
        key="dbpedia_14",
        provider="hf",
        uri="hf:dbpedia_14",
        modality="text",
        task="classification",
        description="DBpedia-14 (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={
            "text_column": "content",
            "label_column": "label",
            "prefer_test_split": True,
        },
    ),
    "yelp_polarity": DatasetSpec(
        key="yelp_polarity",
        provider="hf",
        uri="hf:yelp_polarity",
        modality="text",
        task="classification",
        description="Yelp-2 sentiment (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={"text_column": "text", "label_column": "label", "prefer_test_split": True},
    ),
    "yelp_review_full": DatasetSpec(
        key="yelp_review_full",
        provider="hf",
        uri="hf:yelp_review_full",
        modality="text",
        task="classification",
        description="Yelp-5 reviews (Hugging Face datasets).",
        required_extra="hf",
        source_kwargs={"text_column": "text", "label_column": "label", "prefer_test_split": True},
    ),
}
