from pathlib import Path
import json
import logging
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The uploaded file may have been renamed by the platform.
# The script accepts the canonical name first and common exported variants.
PRODUCT_CANDIDATES = [
    RAW_DIR / "product_info.csv",
    RAW_DIR / "product_info(1).csv",
]

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

# Also accept common platform-renamed filenames.
REVIEW_ALTERNATIVES = {
    "reviews_0-250.csv": ["reviews_0-250(1).csv"],
    "reviews_250-500.csv": ["reviews_250-500(1).csv"],
    "reviews_500-750.csv": ["reviews_500-750(1).csv"],
    "reviews_750-1250.csv": ["reviews_750-1250(1).csv"],
    "reviews_1250-end.csv": ["reviews_1250-end(1).csv"],
}

FIGURES_DIR = PROJECT_ROOT / "reports" / "EDA" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "EDA" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports" / "EDA"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for directory in [FIGURES_DIR, TABLES_DIR, REPORTS_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sephora_eda")


# ============================================================
# HELPERS
# ============================================================

def save_table(df: pd.DataFrame, filename: str) -> None:
    """Save an EDA table as UTF-8 CSV."""
    path = TABLES_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved table: %s", path)


def _csv_safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose pandas StringDtype columns are plain Python objects.

    The project dataset is large (1M+ reviews). With a PyArrow-backed StringDtype,
    pandas can spend a very long time materializing Arrow arrays while writing CSV.
    Converting only string-extension columns to object avoids that bottleneck.
    """
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.StringDtype):
            out[col] = out[col].astype(object)
    return out


def save_processed(df: pd.DataFrame, stem: str) -> None:
    """Save processed data efficiently as CSV and, when possible, Parquet."""
    csv_path = PROCESSED_DIR / f"{stem}.csv"
    csv_df = _csv_safe_copy(df)
    csv_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        chunksize=100_000,
    )
    logger.info("Saved processed CSV: %s", csv_path)
    try:
        parquet_path = PROCESSED_DIR / f"{stem}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved processed Parquet: %s", parquet_path)
    except Exception as exc:
        logger.warning("Parquet not saved for %s: %s", stem, exc)


def save_figure(filename: str) -> None:
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_product_file() -> Path:
    path = first_existing(PRODUCT_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            f"No product_info file found in {RAW_DIR}. "
            f"Expected one of: {[p.name for p in PRODUCT_CANDIDATES]}"
        )
    return path


def find_review_file(canonical_name: str) -> Path:
    candidates = [RAW_DIR / canonical_name]
    candidates.extend(
        RAW_DIR / alt for alt in REVIEW_ALTERNATIVES.get(canonical_name, [])
    )
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            f"Missing review file: {canonical_name}. "
            f"Checked: {[p.name for p in candidates]}"
        )
    return path


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_text_value(value):
    if pd.isna(value):
        return value
    value = str(value).strip()
    return value if value else np.nan


def normalize_categories(series: pd.Series, mapping=None) -> pd.Series:
    """Normalize a large categorical Series using vectorized operations.

    This is intentionally not implemented with Series.map(...): the review
    dataset has 1M+ rows and Python-level callbacks become a major bottleneck.
    """
    s = series.astype("string")
    s = s.str.strip()
    s = s.mask(s.eq(""), pd.NA)
    if mapping:
        s = s.replace(mapping)
    return s


def pct(numerator, denominator):
    return round((numerator / denominator * 100), 4) if denominator else 0.0


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ============================================================
# 1. LOAD
# ============================================================

def load_data():
    product_path = find_product_file()
    logger.info("Loading products from %s", product_path)
    products = pd.read_csv(product_path, low_memory=False)
    products = normalize_column_names(products)

    review_frames = []
    review_metadata = []

    for canonical_name in REVIEW_FILES:
        path = find_review_file(canonical_name)
        logger.info("Loading reviews from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df = normalize_column_names(df)
        df["_source_file"] = path.name
        review_frames.append(df)
        review_metadata.append(
            {
                "file": path.name,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    reviews = pd.concat(review_frames, ignore_index=True, sort=False)

    save_table(pd.DataFrame(review_metadata), "source_files_summary.csv")

    logger.info("Products loaded: %s rows x %s columns", products.shape[0], products.shape[1])
    logger.info("Reviews loaded: %s rows x %s columns", reviews.shape[0], reviews.shape[1])

    return products, reviews


# ============================================================
# 2. INITIAL QUALITY AUDIT
# ============================================================

def dataset_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    total_rows = len(df)

    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "dtype": str(s.dtype),
                "rows": total_rows,
                "missing_count": int(s.isna().sum()),
                "missing_pct": pct(s.isna().sum(), total_rows),
                "unique_count": int(s.nunique(dropna=True)),
                "duplicate_value_count": int(s.duplicated(keep=False).sum()),
                "non_null_count": int(s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def descriptive_statistics(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    result = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    result.insert(0, "dataset", dataset_name)
    return result


def run_initial_quality(products, reviews):
    summary = pd.concat(
        [
            dataset_summary(products, "products_raw"),
            dataset_summary(reviews, "reviews_raw"),
        ],
        ignore_index=True,
    )
    save_table(summary, "dataset_summary.csv")

    save_table(
        descriptive_statistics(products, "products_raw"),
        "products_descriptive_statistics.csv",
    )
    save_table(
        descriptive_statistics(reviews, "reviews_raw"),
        "reviews_descriptive_statistics.csv",
    )

    duplicate_summary = pd.DataFrame(
        [
            {
                "dataset": "products_raw",
                "rows": len(products),
                "exact_duplicate_rows": int(products.duplicated().sum()),
            },
            {
                "dataset": "reviews_raw",
                "rows": len(reviews),
                "exact_duplicate_rows": int(reviews.duplicated().sum()),
            },
        ]
    )
    save_table(duplicate_summary, "duplicate_summary.csv")


# ============================================================
# 3. CLEANING
# ============================================================

def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    # Remove accidental index columns only.
    index_like = [
        c for c in df.columns
        if c.lower().startswith("unnamed:")
    ]
    if index_like:
        df = df.drop(columns=index_like)

    # Standardize textual columns without changing meaning.
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().mask(lambda s: s.eq(""), pd.NA)

    # Numeric conversion where appropriate.
    numeric_candidates = [
        "loves_count", "rating", "reviews", "price_usd",
        "value_price_usd", "sale_price_usd", "child_count",
        "child_max_price", "child_min_price"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean-like columns.
    boolean_candidates = [
        "limited_edition", "new", "online_only",
        "out_of_stock", "sephora_exclusive"
    ]
    for col in boolean_candidates:
        if col in df.columns:
            if df[col].dtype == object:
                mapping = {
                    "true": True, "false": False,
                    "True": True, "False": False,
                    "1": True, "0": False,
                    "yes": True, "no": False,
                    "Yes": True, "No": False,
                }
                df[col] = df[col].replace(mapping)
            df[col] = df[col].astype("boolean")

    # Product IDs are identifiers, never numeric measures.
    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype("string").str.strip().mask(lambda s: s.eq(""), pd.NA)

    # Keep the complete catalog. Only remove exact duplicate rows if present.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    cleaning = pd.DataFrame(
        [{
            "dataset": "products",
            "action": "drop_exact_duplicates",
            "affected_rows": removed,
            "affected_pct": pct(removed, before),
            "justification": "Exact duplicate product records do not add information.",
        }]
    )
    save_table(cleaning, "products_cleaning_decisions.csv")

    return df


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

    # Drop export/index column, not information from the review.
    index_like = [
        c for c in df.columns
        if c.lower().startswith("unnamed:")
    ]
    if index_like:
        df = df.drop(columns=index_like)

    # Keep source for traceability.
    if "_source_file" not in df.columns:
        df["_source_file"] = "unknown"

    # Standardize identifiers.
    for col in ["author_id", "product_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().mask(lambda s: s.eq(""), pd.NA)

    # Numeric columns.
    numeric_candidates = [
        "rating",
        "helpfulness",
        "total_feedback_count",
        "total_neg_feedback_count",
        "total_pos_feedback_count",
        "price_usd",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date.
    if "submission_time" in df.columns:
        df["submission_time"] = pd.to_datetime(
            df["submission_time"], errors="coerce"
        )

    # Text cleanup without replacing missing values with fabricated text.
    text_cols = [
        "review_text", "review_title", "skin_tone",
        "eye_color", "skin_type", "hair_color",
        "product_name", "brand_name"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().mask(lambda s: s.eq(""), pd.NA)

    # Boolean normalization.
    if "is_recommended" in df.columns:
        mapping = {
            "true": True, "false": False,
            "True": True, "False": False,
            "1": True, "0": False,
            "yes": True, "no": False,
            "Yes": True, "No": False,
        }
        if df["is_recommended"].dtype == object:
            df["is_recommended"] = df["is_recommended"].replace(mapping)
        df["is_recommended"] = df["is_recommended"].astype("boolean")

    # Category normalization: only clear semantic duplicates.
    if "eye_color" in df.columns:
        df["eye_color"] = normalize_categories(
            df["eye_color"],
            mapping={
                "Grey": "gray",
                "grey": "gray",
            },
        )

    # Keep rare skin-tone labels unless there is clear evidence they are
    # spelling/format variants. Do not collapse legitimate categories.
    if "skin_tone" in df.columns:
        df["skin_tone"] = normalize_categories(df["skin_tone"])

    if "skin_type" in df.columns:
        df["skin_type"] = normalize_categories(df["skin_type"])

    if "hair_color" in df.columns:
        df["hair_color"] = normalize_categories(df["hair_color"])

    # Remove exact duplicate rows.
    before_exact = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    exact_removed = before_exact - len(df)

    # Remove repeated identical review content by the same author/product.
    # Keep the first record; do not remove every repeated author-product pair.
    subset = [
        c for c in ["author_id", "product_id", "review_text"]
        if c in df.columns
    ]
    if len(subset) == 3:
        before_content = len(df)
        # Do not collapse distinct author/product records merely because all
        # of them have a missing review_text. Only deduplicate when text exists.
        has_text = df["review_text"].notna()
        duplicated_text = df.loc[has_text].duplicated(
            subset=subset, keep="first"
        )
        drop_index = duplicated_text[duplicated_text].index
        df = df.drop(index=drop_index).reset_index(drop=True)
        content_removed = len(drop_index)
    else:
        content_removed = 0

    cleaning = pd.DataFrame(
        [
            {
                "dataset": "reviews",
                "action": "drop_exact_duplicates",
                "affected_rows": exact_removed,
                "affected_pct": pct(exact_removed, before_exact),
                "justification": "Exact duplicate review records do not add information.",
            },
            {
                "dataset": "reviews",
                "action": "drop_duplicate_author_product_review_text",
                "affected_rows": content_removed,
                "affected_pct": pct(content_removed, before_exact),
                "justification": (
                    "The same author/product/review text is treated as a duplicated "
                    "review record; repeated author-product pairs with different text are retained."
                ),
            },
            {
                "dataset": "reviews",
                "action": "normalize_eye_color_gray",
                "affected_rows": np.nan,
                "affected_pct": np.nan,
                "justification": "Grey/gray represent the same eye-color category.",
            },
            {
                "dataset": "reviews",
                "action": "missing_values",
                "affected_rows": np.nan,
                "affected_pct": np.nan,
                "justification": (
                    "Missing user/product attributes are not fabricated. They remain missing "
                    "or are handled explicitly during future feature engineering."
                ),
            },
        ]
    )
    save_table(cleaning, "reviews_cleaning_decisions.csv")

    return df


# ============================================================
# 4. RELATIONSHIP / COVERAGE
# ============================================================

def analyze_product_review_coverage(products, reviews):
    product_ids = products["product_id"].astype(object)
    review_ids = reviews["product_id"].astype(object)

    review_counts = (
        reviews.groupby("product_id", dropna=False)
        .agg(
            review_count=("product_id", "size"),
            unique_users=("author_id", "nunique"),
            average_review_rating=("rating", "mean"),
            recommendation_rate=("is_recommended", "mean"),
        )
        .reset_index()
    )

    coverage = products[
        ["product_id"] +
        [c for c in ["product_name", "brand_name", "primary_category",
                     "secondary_category", "tertiary_category", "price_usd"]
         if c in products.columns]
    ].copy()

    coverage = coverage.merge(
        review_counts,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

    coverage["has_reviews"] = coverage["review_count"].fillna(0).gt(0)
    coverage["review_count"] = coverage["review_count"].fillna(0).astype("int64")
    coverage["unique_users"] = coverage["unique_users"].fillna(0).astype("int64")

    save_table(coverage, "product_review_coverage.csv")
    coverage.to_parquet(PROCESSED_DIR / "product_review_coverage.parquet", index=False)

    # Avoid converting 1M+ Arrow/string values to a Python set.
    # The previous approach was unnecessarily expensive and could hang on large datasets.
    catalog_product_ids = pd.Index(product_ids.dropna().unique())
    review_product_ids = pd.Index(review_ids.dropna().unique())

    matched_product_ids = review_product_ids.intersection(catalog_product_ids)
    unmatched_review_products = review_product_ids.difference(catalog_product_ids)

    relation = pd.DataFrame(
        [{
            "catalog_products": products["product_id"].nunique(),
            "review_product_ids": reviews["product_id"].nunique(),
            "matched_product_ids": len(matched_product_ids),
            "unmatched_review_product_ids": len(unmatched_review_products),
            "products_with_reviews": int(coverage["has_reviews"].sum()),
            "products_without_reviews": int((~coverage["has_reviews"]).sum()),
            "catalog_review_coverage_pct": pct(
                coverage["has_reviews"].sum(), len(coverage)
            ),
            "reviews_with_product_match_pct": pct(
                reviews["product_id"].isin(catalog_product_ids).sum(),
                len(reviews)
            ),
        }]
    )
    save_table(relation, "product_review_relationship.csv")

    if not unmatched_review_products.empty:
        pd.DataFrame({"product_id": unmatched_review_products.astype(object)}).to_csv(
            TABLES_DIR / "unmatched_review_product_ids.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return coverage


def category_analysis(products, reviews):
    """Analyze catalog and review coverage by primary category without merging 1M rows."""
    if "primary_category" not in products.columns:
        return pd.DataFrame()

    category_map = (
        products[["product_id", "primary_category"]]
        .drop_duplicates("product_id")
        .set_index("product_id")["primary_category"]
    )
    review_category = reviews["product_id"].map(category_map).fillna("Unknown")

    product_category = products["primary_category"].fillna("Unknown")
    product_counts = (
        product_category.value_counts()
        .rename_axis("category_for_analysis")
        .reset_index(name="product_count")
    )

    reviewed_products = (
        pd.DataFrame({
            "category_for_analysis": review_category,
            "product_id": reviews["product_id"].values,
        })
        .groupby("category_for_analysis")["product_id"]
        .nunique()
        .rename("products_with_reviews")
        .reset_index()
    )

    stats_df = pd.DataFrame({
        "category_for_analysis": review_category,
        "product_id": reviews["product_id"].values,
        "author_id": reviews["author_id"].values,
        "rating": reviews["rating"].values if "rating" in reviews.columns else np.nan,
        "is_recommended": (
            reviews["is_recommended"].values
            if "is_recommended" in reviews.columns else np.nan
        ),
    })
    review_stats = (
        stats_df.groupby("category_for_analysis")
        .agg(
            review_count=("product_id", "size"),
            unique_users=("author_id", "nunique"),
            average_rating=("rating", "mean"),
            recommendation_rate=("is_recommended", "mean"),
        )
        .reset_index()
    )

    result = (
        product_counts
        .merge(reviewed_products, on="category_for_analysis", how="left")
        .merge(review_stats, on="category_for_analysis", how="left")
    )
    result["products_with_reviews"] = result["products_with_reviews"].fillna(0)
    result["review_count"] = result["review_count"].fillna(0)
    result["unique_users"] = result["unique_users"].fillna(0)
    result["coverage_pct"] = (
        result["products_with_reviews"] / result["product_count"] * 100
    ).round(4)

    result = result.sort_values("product_count", ascending=False)
    save_table(result, "category_analysis.csv")
    return result


# ============================================================
# 5. USER ANALYSIS
# ============================================================

def user_analysis(reviews):
    required = {"author_id"}
    if not required.issubset(reviews.columns):
        return pd.DataFrame()

    user_counts = (
        reviews.groupby("author_id")
        .agg(
            review_count=("product_id", "size"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    profile_cols = [
        c for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]
        if c in reviews.columns
    ]

    if profile_cols:
        # Vectorized aggregation: much faster than applying a Python lambda
        # once per user/column. With ~500k users, the previous approach could
        # take several minutes and appear to hang.
        profile_presence = (
            reviews[profile_cols]
            .notna()
            .groupby(reviews["author_id"], sort=False)
            .any()
        )
        profile_presence["profile_complete"] = profile_presence.all(axis=1)
        user_counts = user_counts.merge(
            profile_presence[["profile_complete"]].reset_index(),
            on="author_id",
            how="left",
        )
    else:
        user_counts["profile_complete"] = False

    save_table(user_counts, "user_interaction_summary.csv")

    summary = pd.DataFrame(
        [{
            "unique_users": reviews["author_id"].nunique(),
            "users_with_one_review": int((user_counts["review_count"] == 1).sum()),
            "users_with_2plus_reviews": int((user_counts["review_count"] >= 2).sum()),
            "users_with_10plus_reviews": int((user_counts["review_count"] >= 10).sum()),
            "profile_complete_users": int(user_counts["profile_complete"].sum()),
            "profile_complete_pct": pct(
                user_counts["profile_complete"].sum(), len(user_counts)
            ),
        }]
    )
    save_table(summary, "user_summary.csv")
    return user_counts


def build_user_profiles(reviews):
    """Build one profile row per user using vectorized operations.

    Categorical profile values are selected by frequency (mode) without
    Python-level groupby functions, which is important for >500k users.
    Missing values are ignored; users with no observed value for a field
    keep NaN rather than receiving an artificial imputation.
    """
    if "author_id" not in reviews.columns:
        return pd.DataFrame()

    profile_cols = [
        c for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]
        if c in reviews.columns
    ]

    # Start with the complete user universe.
    profiles = reviews[["author_id"]].drop_duplicates().copy()

    # Add the most frequent observed value for each categorical attribute.
    # This avoids groupby(...).agg(lambda ...) over hundreds of thousands
    # of groups, which is extremely slow in large datasets.
    for col in profile_cols:
        observed = reviews[["author_id", col]].dropna(subset=[col])
        if observed.empty:
            profiles[col] = np.nan
            continue

        frequencies = (
            observed.groupby(["author_id", col], sort=False)
            .size()
            .reset_index(name="frequency")
        )

        # Deterministic tie-breaking: highest frequency first, then the
        # lexical value of the category.
        frequencies[col] = frequencies[col].astype(str)
        frequencies = frequencies.sort_values(
            ["author_id", "frequency", col],
            ascending=[True, False, True],
            kind="mergesort",
        )

        top_values = frequencies.drop_duplicates("author_id")[
            ["author_id", col]
        ]
        profiles = profiles.merge(top_values, on="author_id", how="left")

    profiles.to_parquet(PROCESSED_DIR / "user_profiles.parquet", index=False)
    _csv_safe_copy(profiles).to_csv(
        PROCESSED_DIR / "user_profiles.csv",
        index=False,
        encoding="utf-8-sig",
        chunksize=100_000,
    )
    logger.info("User profiles created: %s users", len(profiles))
    return profiles


# ============================================================
# 6. INTERACTIONS
# ============================================================

def build_interactions(reviews):
    interaction_cols = [
        c for c in [
            "author_id", "product_id", "rating",
            "is_recommended", "submission_time",
            "helpfulness", "total_feedback_count",
            "total_pos_feedback_count", "total_neg_feedback_count"
        ] if c in reviews.columns
    ]

    interactions = reviews[interaction_cols].copy()

    # Keep only rows with identifiers needed to define an interaction.
    interactions = interactions.dropna(
        subset=[c for c in ["author_id", "product_id"] if c in interactions.columns]
    )

    interactions.to_parquet(
        PROCESSED_DIR / "interactions.parquet", index=False
    )
    _csv_safe_copy(interactions).to_csv(
        PROCESSED_DIR / "interactions.csv",
        index=False,
        encoding="utf-8-sig",
        chunksize=100_000,
    )
    return interactions


# ============================================================
# 7. CLEAN PRODUCT / REVIEW EXPORTS
# ============================================================

def export_clean_datasets(products, reviews):
    """Persist the cleaned datasets without forcing Arrow strings through CSV.

    Parquet is written first because it is the efficient working format. CSV is
    also produced for portability, in chunks, using object-backed strings.
    """
    products.to_parquet(PROCESSED_DIR / "products_clean.parquet", index=False)
    reviews.to_parquet(PROCESSED_DIR / "reviews_clean.parquet", index=False)
    logger.info("Saved cleaned Parquet datasets")

    products_csv = _csv_safe_copy(products)
    reviews_csv = _csv_safe_copy(reviews)

    products_csv.to_csv(
        PROCESSED_DIR / "products_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logger.info("Saved products_clean.csv")

    reviews_csv.to_csv(
        PROCESSED_DIR / "reviews_clean.csv",
        index=False,
        encoding="utf-8-sig",
        chunksize=100_000,
    )
    logger.info("Saved reviews_clean.csv")


# ============================================================
# 8. EDA TABLES
# ============================================================

def missing_analysis(products, reviews):
    frames = []
    for name, df in [("products_clean", products), ("reviews_clean", reviews)]:
        temp = dataset_summary(df, name)
        frames.append(temp)

    result = pd.concat(frames, ignore_index=True)
    save_table(result, "missing_values.csv")
    return result


def outlier_analysis(products, reviews):
    rows = []

    # Outliers are audited, not automatically deleted. In e-commerce, high prices
    # and high feedback counts may be valid long-tail observations.
    for dataset_name, df in [("products", products), ("reviews", reviews)]:
        for col in ["price_usd", "rating", "total_feedback_count", "total_pos_feedback_count", "helpfulness"]:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
            rows.append({
                "dataset": dataset_name,
                "variable": col,
                "n_non_null": int(s.size),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": low,
                "upper_bound": high,
                "n_iqr_outliers": int(mask.sum()),
                "pct_iqr_outliers": float(mask.mean() * 100),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        save_table(result, "outlier_audit.csv")


def rating_analysis(reviews):
    if "rating" not in reviews.columns:
        return pd.DataFrame()

    dist = (
        reviews["rating"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="review_count")
    )
    dist["percentage"] = (
        dist["review_count"] / len(reviews) * 100
    ).round(4)
    save_table(dist, "rating_distribution.csv")

    if "primary_category" in reviews.columns:
        by_cat = (
            reviews.groupby("primary_category")
            .agg(
                review_count=("rating", "size"),
                average_rating=("rating", "mean"),
                median_rating=("rating", "median"),
            )
            .reset_index()
        )
        save_table(by_cat, "rating_by_category.csv")

    return dist


def recommendation_analysis(reviews):
    if "is_recommended" not in reviews.columns:
        return pd.DataFrame()

    dist = (
        reviews["is_recommended"]
        .value_counts(dropna=False)
        .rename_axis("is_recommended")
        .reset_index(name="review_count")
    )
    dist["percentage"] = (
        dist["review_count"] / len(reviews) * 100
    ).round(4)
    save_table(dist, "recommendation_distribution.csv")
    return dist


def product_analysis(products, reviews):
    stats = (
        reviews.groupby("product_id")
        .agg(
            review_count=("product_id", "size"),
            unique_users=("author_id", "nunique"),
            average_rating=("rating", "mean"),
            recommendation_rate=("is_recommended", "mean"),
        )
        .reset_index()
        .sort_values("review_count", ascending=False)
    )

    if "primary_category" in products.columns:
        stats = stats.merge(
            products[["product_id", "primary_category", "product_name", "brand_name"]],
            on="product_id",
            how="left",
        )

    save_table(stats, "product_interaction_analysis.csv")
    save_table(stats.head(100), "top_100_products_by_reviews.csv")
    return stats


def interaction_sparsity(reviews):
    users = reviews["author_id"].nunique()
    products = reviews["product_id"].nunique()
    interactions = len(reviews)

    possible = users * products
    density = interactions / possible if possible else 0
    sparsity = 1 - density

    result = pd.DataFrame(
        [{
            "unique_users": users,
            "unique_products_with_reviews": products,
            "observed_interactions": interactions,
            "possible_user_product_pairs": possible,
            "density_pct": density * 100,
            "sparsity_pct": sparsity * 100,
        }]
    )
    save_table(result, "sparsity_analysis.csv")
    return result


def cold_start_analysis(products, reviews, coverage):
    user_counts = reviews.groupby("author_id").size()

    result = pd.DataFrame(
        [{
            "catalog_products": len(products),
            "products_without_reviews": int((~coverage["has_reviews"]).sum()),
            "products_without_reviews_pct": pct(
                (~coverage["has_reviews"]).sum(), len(coverage)
            ),
            "users_with_one_interaction": int((user_counts == 1).sum()),
            "users_with_one_interaction_pct": pct(
                (user_counts == 1).sum(), len(user_counts)
            ),
        }]
    )
    save_table(result, "cold_start_analysis.csv")
    return result


# ============================================================
# 9. PLOTS
# ============================================================

def plot_category_products(products):
    if "primary_category" not in products.columns:
        return
    data = (
        products["primary_category"]
        .fillna("Unknown")
        .value_counts()
        .sort_values(ascending=True)
    )
    plt.figure(figsize=(10, 6))
    data.plot(kind="barh")
    plt.title("Products by Primary Category")
    plt.xlabel("Number of products")
    plt.ylabel("Category")
    save_figure("01_products_by_category.png")


def plot_product_review_coverage(coverage):
    data = coverage["has_reviews"].value_counts()
    labels = ["With reviews" if x else "Without reviews" for x in data.index]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, data.values)
    plt.title("Products With vs Without Reviews")
    plt.ylabel("Number of products")
    save_figure("02_products_with_without_reviews.png")


def plot_category_coverage(category_df):
    if category_df.empty:
        return
    data = category_df.sort_values("coverage_pct", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data["category_for_analysis"].astype(str), data["coverage_pct"])
    plt.title("Review Coverage by Category")
    plt.xlabel("Products with reviews (%)")
    plt.ylabel("Category")
    save_figure("03_review_coverage_by_category.png")


def plot_reviews_by_category(reviews, products):
    if "primary_category" not in products.columns:
        return
    category_map = products.drop_duplicates("product_id").set_index("product_id")["primary_category"]
    data = (
        reviews["product_id"].map(category_map).fillna("Unknown")
        .value_counts()
        .sort_values(ascending=True)
    )
    plt.figure(figsize=(10, 6))
    plt.barh(data.index.astype(str), data.values)
    plt.title("Reviews by Product Category")
    plt.xlabel("Number of reviews")
    plt.ylabel("Category")
    save_figure("04_reviews_by_category.png")


def plot_rating_distribution(reviews):
    if "rating" not in reviews.columns:
        return
    data = reviews["rating"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(data.index.astype(str), data.values)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of reviews")
    save_figure("05_rating_distribution.png")


def plot_rating_by_category(reviews, products):
    if "primary_category" not in products.columns or "rating" not in reviews.columns:
        return
    category_map = products.drop_duplicates("product_id").set_index("product_id")["primary_category"]
    temp = pd.DataFrame({
        "category": reviews["product_id"].map(category_map).fillna("Unknown"),
        "rating": reviews["rating"].values,
    })
    data = temp.groupby("category")["rating"].mean().sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data.index.astype(str), data.values)
    plt.title("Average Rating by Category")
    plt.xlabel("Average rating")
    plt.ylabel("Category")
    save_figure("06_average_rating_by_category.png")


def plot_recommendation_distribution(reviews):
    if "is_recommended" not in reviews.columns:
        return
    data = reviews["is_recommended"].value_counts(dropna=False)
    labels = [
        "Recommended" if x is True
        else "Not recommended" if x is False
        else "Missing"
        for x in data.index
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, data.values)
    plt.title("Recommendation Flag Distribution")
    plt.ylabel("Number of reviews")
    save_figure("07_recommendation_distribution.png")


def plot_profile_completeness(reviews):
    profile_cols = [c for c in ["skin_type", "skin_tone", "eye_color", "hair_color"] if c in reviews.columns]
    if not profile_cols or "author_id" not in reviews.columns:
        return

    presence = reviews[profile_cols].notna()
    presence["author_id"] = reviews["author_id"].values
    user_profile = presence.groupby("author_id", sort=False)[profile_cols].any()
    completeness = user_profile.sum(axis=1).value_counts().sort_index()
    labels = [f"{i}/{len(profile_cols)} atributos" for i in completeness.index]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, completeness.values)
    plt.title("Completitud del perfil por usuario")
    plt.xlabel("Atributos de perfil disponibles")
    plt.ylabel("Número de usuarios")
    save_figure("profile_completeness_users.png")


def plot_profile_distributions(reviews):
    profile_cols = [
        ("skin_type", "10_skin_type.png"),
        ("skin_tone", "11_skin_tone.png"),
        ("eye_color", "12_eye_color.png"),
        ("hair_color", "13_hair_color.png"),
    ]

    for col, filename in profile_cols:
        if col not in reviews.columns:
            continue
        data = reviews[col].fillna("Unknown").value_counts().head(20)
        data = data.sort_values(ascending=True)
        plt.figure(figsize=(10, 7))
        plt.barh(data.index.astype(str), data.values)
        plt.title(f"{col} Distribution")
        plt.xlabel("Number of reviews")
        plt.ylabel(col)
        save_figure(filename)


def plot_missing_values(products, reviews):
    for df, name in [(products, "products"), (reviews, "reviews")]:
        missing = (df.isna().mean() * 100).sort_values(ascending=True)
        missing = missing[missing > 0]
        if missing.empty:
            continue
        plt.figure(figsize=(10, max(5, len(missing) * 0.28)))
        plt.barh(missing.index.astype(str), missing.values)
        plt.title(f"Missing Values - {name}")
        plt.xlabel("Missing (%)")
        plt.ylabel("Column")
        save_figure(f"14_missing_values_{name}.png")


def plot_price_distribution(products):
    if "price_usd" not in products.columns:
        return
    data = pd.to_numeric(products["price_usd"], errors="coerce").dropna()
    data = data[data >= 0]
    plt.figure(figsize=(9, 5))
    plt.hist(data, bins=50)
    plt.title("Product Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Number of products")
    save_figure("16_price_distribution.png")


def plot_long_tail_products(product_stats):
    data = product_stats["review_count"].sort_values(ascending=False).reset_index(drop=True)
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(data) + 1), data.values)
    plt.yscale("log")
    plt.xlabel("Products ranked by number of reviews")
    plt.ylabel("Reviews (log scale)")
    plt.title("Long Tail of Product Interactions")
    save_figure("17_product_long_tail.png")


def plot_long_tail_users(reviews):
    data = (
        reviews.groupby("author_id")
        .size()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(data) + 1), data.values)
    plt.yscale("log")
    plt.xlabel("Users ranked by number of reviews")
    plt.ylabel("Reviews (log scale)")
    plt.title("Long Tail of User Interactions")
    save_figure("18_user_long_tail.png")


def plot_reviews_vs_rating(product_stats):
    if product_stats.empty:
        return
    plt.figure(figsize=(9, 6))
    plt.scatter(
        product_stats["review_count"],
        product_stats["average_rating"],
        alpha=0.35,
        s=12,
    )
    plt.xscale("log")
    plt.xlabel("Number of reviews (log scale)")
    plt.ylabel("Average rating")
    plt.title("Review Count vs Average Rating")
    save_figure("19_reviews_vs_average_rating.png")


def plot_sparsity(sparsity_df):
    row = sparsity_df.iloc[0]
    values = [row["density_pct"], row["sparsity_pct"]]
    labels = ["Density", "Sparsity"]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("User-Product Interaction Density vs Sparsity")
    plt.ylabel("Percentage")
    save_figure("20_sparsity.png")


def plot_top_products(product_stats):
    if product_stats.empty:
        return
    data = product_stats.head(20).sort_values("review_count", ascending=True)
    labels = (
        data["product_name"].fillna(data["product_id"]).astype(str)
        if "product_name" in data.columns
        else data["product_id"].astype(str)
    )
    plt.figure(figsize=(10, 8))
    plt.barh(labels, data["review_count"])
    plt.title("Top 20 Products by Review Count")
    plt.xlabel("Reviews")
    plt.ylabel("Product")
    save_figure("21_top_20_products.png")


# ============================================================
# 10. DATA DICTIONARY
# ============================================================

def build_data_dictionary(products, reviews):
    rows = []

    descriptions = {
        "product_id": "Unique product identifier.",
        "product_name": "Product name.",
        "brand_name": "Brand associated with the product.",
        "primary_category": "Primary product category.",
        "secondary_category": "Secondary product category.",
        "tertiary_category": "Tertiary product category.",
        "price_usd": "Product price in USD.",
        "rating": "User rating on the review scale.",
        "is_recommended": "Whether the reviewer recommended the product.",
        "author_id": "User/reviewer identifier.",
        "skin_type": "Self-reported skin type.",
        "skin_tone": "Self-reported skin tone.",
        "eye_color": "Self-reported eye color.",
        "hair_color": "Self-reported hair color.",
        "submission_time": "Review submission timestamp.",
        "review_text": "Review text.",
        "review_title": "Review title.",
        "helpfulness": "Review helpfulness measure.",
    }

    for dataset_name, df in [
        ("products_clean", products),
        ("reviews_clean", reviews),
    ]:
        for col in df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "description": descriptions.get(col, "Dataset field."),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_pct": pct(df[col].isna().sum(), len(df)),
                    "unique_count": int(df[col].nunique(dropna=True)),
                }
            )

    result = pd.DataFrame(rows)
    save_table(result, "data_dictionary.csv")
    return result


# ============================================================
# 11. REPORT
# ============================================================

def generate_html_report(products, reviews, coverage, category_df, sparsity_df):
    product_count = len(products)
    review_count = len(reviews)
    user_count = reviews["author_id"].nunique()
    reviewed_products = int(coverage["has_reviews"].sum())
    no_review_products = int((~coverage["has_reviews"]).sum())
    review_coverage = pct(reviewed_products, product_count)

    complete_profile_cols = [
        c for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]
        if c in reviews.columns
    ]
    if complete_profile_cols:
        profile_complete_pct = (
            reviews[complete_profile_cols].notna().all(axis=1).mean() * 100
        )
    else:
        profile_complete_pct = np.nan

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sephora EDA Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; }}
h1, h2 {{ color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f2f2f2; }}
.metric {{ display: inline-block; padding: 15px; margin: 8px; border: 1px solid #ddd; }}
img {{ max-width: 900px; margin: 10px 0; }}
</style>
</head>
<body>

<h1>Sephora — Exploratory Data Analysis</h1>

<h2>1. Objective</h2>
<p>
This EDA prepares the Sephora catalog and review data for a future personalized
product recommendation system. The current scope preserves the complete catalog
rather than filtering products to a single category.
</p>

<h2>2. Dataset overview</h2>

<div class="metric"><b>Catalog products</b><br>{product_count:,}</div>
<div class="metric"><b>Reviews</b><br>{review_count:,}</div>
<div class="metric"><b>Unique users</b><br>{user_count:,}</div>
<div class="metric"><b>Products with reviews</b><br>{reviewed_products:,}</div>
<div class="metric"><b>Products without reviews</b><br>{no_review_products:,}</div>
<div class="metric"><b>Catalog review coverage</b><br>{review_coverage:.2f}%</div>
<div class="metric"><b>Complete profile rows</b><br>{profile_complete_pct:.2f}%</div>

<h2>3. Main quality decisions</h2>
<ul>
<li>Raw files are preserved and are not overwritten.</li>
<li>All product categories are retained.</li>
<li>Products without reviews are retained to preserve the complete catalog and expose cold-start behavior.</li>
<li>Exact duplicate records are removed from processed data.</li>
<li>Repeated identical author/product/review-text records are treated as duplicate review records.</li>
<li>Missing values are not replaced with fabricated information.</li>
<li>Only clear semantic category variants are normalized.</li>
</ul>

<h2>4. Main recommendation-system risks</h2>
<ul>
<li>Sparsity of the user-product interaction space.</li>
<li>Long-tail distribution of product and user interactions.</li>
<li>Cold start for products without review history.</li>
<li>Cold start / weak history for users with very few interactions.</li>
<li>Incomplete user profile attributes.</li>
<li>Potentially positive-skewed ratings and recommendation labels.</li>
</ul>

<h2>5. Visualizations</h2>
"""

    for image_path in sorted(FIGURES_DIR.glob("*.png")):
        rel = image_path.relative_to(REPORTS_DIR.parent).as_posix()
        html += f"<h3>{image_path.stem}</h3><img src='figures/{image_path.name}' alt='{image_path.stem}'>"

    html += """
</body>
</html>
"""

    report_path = REPORTS_DIR / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Saved report: %s", report_path)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

def save_pipeline_summary(products, reviews, coverage, sparsity_df):
    summary = {
        "raw_and_processed_catalog_products": int(len(products)),
        "processed_reviews": int(len(reviews)),
        "unique_users": int(reviews["author_id"].nunique()),
        "products_with_reviews": int(coverage["has_reviews"].sum()),
        "products_without_reviews": int((~coverage["has_reviews"]).sum()),
        "catalog_review_coverage_pct": pct(
            coverage["has_reviews"].sum(), len(coverage)
        ),
        "sparsity_pct": float(sparsity_df.iloc[0]["sparsity_pct"]),
        "density_pct": float(sparsity_df.iloc[0]["density_pct"]),
    }

    path = TABLES_DIR / "pipeline_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_safe),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("Catalog products: %s", summary["raw_and_processed_catalog_products"])
    logger.info("Processed reviews: %s", summary["processed_reviews"])
    logger.info("Unique users: %s", summary["unique_users"])
    logger.info("Products with reviews: %s", summary["products_with_reviews"])
    logger.info("Products without reviews: %s", summary["products_without_reviews"])
    logger.info("Review coverage: %.2f%%", summary["catalog_review_coverage_pct"])
    logger.info("Sparsity: %.4f%%", summary["sparsity_pct"])
    logger.info("Reports directory: %s", REPORTS_DIR)
    logger.info("Processed data directory: %s", PROCESSED_DIR)
    logger.info("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    logger.info("Starting Sephora EDA pipeline")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Database directory: %s", RAW_DIR)

    products_raw, reviews_raw = load_data()

    # Initial audit BEFORE transformations.
    run_initial_quality(products_raw, reviews_raw)

    # Cleaning.
    products = clean_products(products_raw)
    reviews = clean_reviews(reviews_raw)

    # Export core cleaned datasets.
    export_clean_datasets(products, reviews)

    # Relationships and derived datasets.
    coverage = analyze_product_review_coverage(products, reviews)
    category_df = category_analysis(products, reviews)

    users = user_analysis(reviews)
    user_profiles = build_user_profiles(reviews)
    interactions = build_interactions(reviews)

    # Export derived datasets used later by modeling.
    save_processed(coverage, "product_review_coverage")
    save_processed(user_profiles, "user_profiles")
    save_processed(interactions, "interactions")
    save_processed(category_df, "category_analysis")

    # EDA tables.
    missing_analysis(products, reviews)
    outlier_analysis(products, reviews)
    rating_analysis(reviews)
    recommendation_analysis(reviews)
    product_stats = product_analysis(products, reviews)
    sparsity_df = interaction_sparsity(reviews)
    cold_start_analysis(products, reviews, coverage)
    build_data_dictionary(products, reviews)

    # Figures.
    plot_category_products(products)
    plot_product_review_coverage(coverage)
    plot_category_coverage(category_df)
    plot_reviews_by_category(reviews, products)
    plot_rating_distribution(reviews)
    plot_rating_by_category(reviews, products)
    plot_recommendation_distribution(reviews)
    plot_profile_completeness(reviews)
    plot_profile_distributions(reviews)
    plot_missing_values(products, reviews)
    plot_price_distribution(products)
    plot_long_tail_products(product_stats)
    plot_long_tail_users(reviews)
    plot_reviews_vs_rating(product_stats)
    plot_sparsity(sparsity_df)
    plot_top_products(product_stats)

    # Report and final summary.
    generate_html_report(products, reviews, coverage, category_df, sparsity_df)
    save_pipeline_summary(products, reviews, coverage, sparsity_df)

    logger.info("All requested EDA and preparation outputs were generated.")


if __name__ == "__main__":
    main()
