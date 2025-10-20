"""Topic modeling for translated Udhyam chat messages using BERTopic.

Messages are in English after translation from Hindi, Punjabi, and Hinglish.
This version uses improved preprocessing, better embeddings, and tuned parameters
for clearer, more distinct topics.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from bertopic import BERTopic  # type: ignore
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependencies for topic modeling.\n"
        "Install with: pip install bertopic[visualization] sentence-transformers umap-learn hdbscan"
    ) from exc

try:
    from bertopic.representation import OpenAI as OpenAIRepresentation  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAIRepresentation = None  # type: ignore

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


DEFAULT_INPUT = "data/cleaned/messages_translated.csv"
OUTPUT_BASE = Path("data/analysis")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Translated CSV input")
    parser.add_argument("--min-topic-size", type=int, default=25, help="Min messages per topic (increased for quality)")
    parser.add_argument("--min-cluster-size", type=int, default=25, help="HDBSCAN min cluster size")
    parser.add_argument("--top-n", type=int, default=10, help="Top keywords per topic")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.05, help="UMAP min_dist parameter")
    parser.add_argument("--reduce-topics", action="store_true", help="Auto-reduce topics after fitting")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit per role for debugging",
    )
    parser.add_argument(
        "--use-openai-representation",
        action="store_true",
        help="Use OpenAI chat models to refine topic representations",
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_TOPIC_MODEL", "gpt-4o-mini"),
        help="OpenAI model name for topic representations",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for topic refinement prompts",
    )
    return parser.parse_args(argv)


def ensure_nltk_data() -> None:
    """Download required NLTK data for stopwords and lemmatization."""
    for resource in ["corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"]:
        try:
            nltk.data.find(resource)
        except LookupError:  # pragma: no cover - first-time setup
            resource_name = resource.split("/")[-1]
            nltk.download(resource_name, quiet=True)


def ensure_output_dirs() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def build_stopwords() -> list[str]:
    """Build comprehensive stopword list for English (translated from Hindi/Punjabi/Hinglish)."""
    ensure_nltk_data()
    english = set(stopwords.words("english"))

    # Platform-specific and repetitive terms to remove
    platform_noise = {
        "kbc",
        "portal",
        "please",
        "clarify",
        "please clarify",
        "thanks",
        "thank",
        "hello",
        "hi",
        "hey",
        "okay",
        "ok",
        "yes",
        "no",
        "yeah",
    }

    # Common filler words
    fillers = {
        "eh",
        "ah",
        "uh",
        "mm",
        "mmm",
        "hm",
        "hmm",
        "um",
        "umm",
    }

    return list(english | platform_noise | fillers)


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better topic modeling.

    Removes URLs, IDs, platform-specific noise, and applies lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers and IDs (sequences of digits)
    text = re.sub(r'\b\d+\b', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower().strip()

    return text


def lemmatize_text(text: str) -> str:
    """Apply lemmatization to reduce words to their base form."""
    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)


def merge_common_phrases(text: str) -> str:
    """Merge common multi-word phrases into single tokens for better topic coherence."""
    # Define common phrases to merge (underscore-separated)
    phrase_replacements = {
        r'\bpeople planet profit\b': 'people_planet_profit',
        r'\btriple bottom line\b': 'triple_bottom_line',
        r'\bentrepreneurial mindset\b': 'entrepreneurial_mindset',
        r'\bbusiness model\b': 'business_model',
        r'\bsocial impact\b': 'social_impact',
        r'\bvalue proposition\b': 'value_proposition',
        r'\bcustomer segment\b': 'customer_segment',
        r'\brevenue stream\b': 'revenue_stream',
        r'\bcost structure\b': 'cost_structure',
        r'\bkey resource\b': 'key_resource',
        r'\bkey activity\b': 'key_activity',
        r'\bkey partner\b': 'key_partner',
        r'\bproblem solving\b': 'problem_solving',
        r'\bcritical thinking\b': 'critical_thinking',
    }

    for pattern, replacement in phrase_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


def clean_message(text: str) -> str:
    """Apply full preprocessing pipeline to a message."""
    text = preprocess_text(text)
    text = merge_common_phrases(text)
    text = lemmatize_text(text)
    return text


def create_topic_model(
    stop_words: list[str],
    min_topic_size: int,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    min_cluster_size: int = 25,
    representation_model: Optional[object] = None,
) -> BERTopic:
    """Create improved BERTopic model with better embeddings and tuned parameters.

    Uses paraphrase-multilingual-MiniLM-L12-v2 for embeddings (good for mixed
    English-Indian languages), with tuned UMAP and HDBSCAN for better separation.

    Args:
        stop_words: List of stopwords to exclude
        min_topic_size: Minimum messages required per topic
        n_neighbors: UMAP n_neighbors (higher = more global structure)
        min_dist: UMAP min_dist (lower = tighter clusters)
        min_cluster_size: HDBSCAN minimum cluster size

    Returns:
        Configured BERTopic model
    """
    # Use multilingual model optimized for paraphrases (good for translated text)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Tune UMAP for better topic separation
    umap_model = UMAP(
        n_neighbors=n_neighbors,  # Balance between local and global structure
        n_components=5,  # Dimensionality of reduced space
        min_dist=min_dist,  # Minimum distance between points (tighter clusters)
        metric='cosine',  # Good for text embeddings
        random_state=42,
    )

    # Tune HDBSCAN for clearer clusters
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,  # Larger clusters = fewer, clearer topics
        min_samples=5,  # More conservative clustering
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass (better for varied densities)
        prediction_data=True,
    )

    # Configure vectorizer for keyword extraction
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 3),  # Include trigrams for phrase capture
        min_df=3,  # Must appear in at least 3 documents
        max_df=0.8,  # Remove if appears in >80% of documents
    )

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        representation_model=representation_model,
        calculate_probabilities=False,  # Faster computation
        verbose=True,  # Show progress
    )


def build_openai_representation(enable: bool, model_name: str, temperature: float) -> Optional[object]:
    """Create an OpenAI-based representation model when requested.

    The OpenAI representation model refines raw topic keywords into human-readable
    topic labels using GPT models. This process occurs after clustering:

    1. Initial topic extraction: BERTopic uses c-TF-IDF to extract keywords from each cluster
    2. OpenAI refinement: The keywords and sample documents are sent to the OpenAI API
    3. LLM interpretation: The model generates concise, interpretable topic labels
    4. Final representation: These labels replace or supplement the raw keywords

    This approach produces more coherent, natural-language topic descriptions compared
    to raw keyword lists, making topics easier to interpret for non-technical users.

    The temperature parameter controls randomness (0.0 = deterministic, 1.0 = creative).
    Lower values are recommended for consistent, factual topic labels.

    Args:
        enable: Whether to use OpenAI representation
        model_name: OpenAI model to use (e.g., 'gpt-4o-mini', 'gpt-4')
        temperature: Sampling temperature for generation (0.0-1.0)

    Returns:
        OpenAI representation model instance or None if disabled
    """

    if not enable:
        return None

    if openai is None or OpenAIRepresentation is None:
        raise SystemExit(
            "OpenAI topic representations require the 'openai' package and BERTopic's OpenAI extras. "
            "Install with: pip install 'openai>=1.0.0' bertopic[openai]"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set to use OpenAI topic representations.")

    client = openai.OpenAI(api_key=api_key)
    return OpenAIRepresentation(client, model=model_name, chat=True, temperature=temperature)


def compute_topic_metrics(topic_model: BERTopic, texts: list[str]) -> dict:
    """Compute coherence and diversity metrics for topic quality assessment.

    Args:
        topic_model: Fitted BERTopic model
        texts: Original text documents

    Returns:
        Dictionary with coherence and diversity scores
    """
    topics_info = topic_model.get_topic_info()
    topics_info = topics_info[topics_info.Topic != -1]

    if topics_info.empty:
        return {"num_topics": 0, "diversity": 0.0, "avg_topic_size": 0}

    # Calculate topic diversity (uniqueness of top words across topics)
    all_top_words = set()
    total_top_words = 0

    for topic_id in topics_info.Topic.tolist():
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            top_10_words = [word for word, _ in topic_words[:10]]
            all_top_words.update(top_10_words)
            total_top_words += len(top_10_words)

    diversity = len(all_top_words) / total_top_words if total_top_words > 0 else 0.0

    metrics = {
        "num_topics": len(topics_info),
        "diversity": round(diversity, 3),
        "avg_topic_size": int(topics_info.Count.mean()),
        "median_topic_size": int(topics_info.Count.median()),
        "total_messages": len(texts),
        "outliers": int(topics_info[topics_info.Topic == -1].Count.sum()) if -1 in topics_info.Topic.values else 0,
    }

    return metrics


def summarise_topics(topic_model: BERTopic, role: str, top_n: int) -> pd.DataFrame:
    """Extract topic information with keywords and message counts.

    Args:
        topic_model: Fitted BERTopic model
        role: Message role (user or assistant)
        top_n: Number of top keywords to extract per topic

    Returns:
        DataFrame with topic summaries
    """
    info = topic_model.get_topic_info()
    info = info[info.Topic != -1].copy()
    info.insert(0, "message_role", role)
    info.rename(columns={"Name": "keywords", "Count": "message_count", "Topic": "topic_id"}, inplace=True)

    rows = []
    for topic_id in info["topic_id"].tolist():
        terms = topic_model.get_topic(topic_id)
        if not terms:
            continue
        keywords = ", ".join(word for word, _ in terms[:top_n])
        rows.append({
            "message_role": role,
            "topic_id": topic_id,
            "keywords": keywords,
            "message_count": int(info.loc[info.topic_id == topic_id, "message_count"].iloc[0]),
        })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_output_dirs()

    print("Loading data...")
    df = pd.read_csv(args.input)

    print("Building stopwords...")
    stop_words = build_stopwords()

    results = []
    topic_keywords_frames: list[pd.DataFrame] = []
    all_metrics = []

    role_columns = {
        "user": "user_msg_en",
        "assistant": "ai_msg_en",
    }

    for role, column in role_columns.items():
        if column not in df.columns:
            print(f"⚠ Column {column} not found, skipping {role}...")
            continue

        # Extract and filter texts
        texts = df[column].fillna("").astype(str)
        mask = texts.str.strip() != ""
        texts = texts[mask]

        if args.max_docs is not None:
            texts = texts.head(args.max_docs)

        if texts.empty:
            print(f"⚠ No messages found for {role}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {role} messages ({len(texts)} total)")
        print(f"{'='*60}")

        # Apply preprocessing pipeline
        print("Preprocessing texts (cleaning, lemmatization, phrase merging)...")
        cleaned_texts = [clean_message(text) for text in texts.tolist()]

        # Filter out empty messages after cleaning
        cleaned_pairs = [(original, cleaned) for original, cleaned in zip(texts.tolist(), cleaned_texts) if cleaned.strip()]
        if not cleaned_pairs:
            print(f"⚠ No valid messages after preprocessing for {role}, skipping...")
            continue

        original_texts, cleaned_texts = zip(*cleaned_pairs)
        print(f"✓ {len(cleaned_texts)} messages after preprocessing")

        # Create and fit model
        print("Creating BERTopic model with improved parameters...")
        representation_model = build_openai_representation(
            args.use_openai_representation,
            args.openai_model,
            args.openai_temperature,
        )
        if representation_model:
            print(f"✓ OpenAI representation enabled ({args.openai_model})")
        topic_model = create_topic_model(
            stop_words,
            args.min_topic_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            min_cluster_size=args.min_cluster_size,
            representation_model=representation_model,
        )

        print("Fitting topics...")
        topics, _ = topic_model.fit_transform(list(cleaned_texts))

        # Get initial topic count
        initial_topics = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"✓ Initial topics found: {initial_topics}")

        # Apply topic reduction if requested
        if args.reduce_topics and initial_topics > 0:
            print("Applying automatic topic reduction...")
            topic_model.reduce_topics(list(cleaned_texts), nr_topics="auto")
            topics = topic_model.topics_
            final_topics = len(set(topics)) - (1 if -1 in topics else 0)
            print(f"✓ Topics after reduction: {final_topics} (reduced by {initial_topics - final_topics})")

        # Compute metrics
        print("Computing topic quality metrics...")
        metrics = compute_topic_metrics(topic_model, list(cleaned_texts))
        metrics["message_role"] = role
        all_metrics.append(metrics)

        print(f"\nTopic Quality Metrics for {role}:")
        print(f"  - Number of topics: {metrics['num_topics']}")
        print(f"  - Topic diversity: {metrics['diversity']:.3f}")
        print(f"  - Avg topic size: {metrics['avg_topic_size']}")
        print(f"  - Median topic size: {metrics['median_topic_size']}")
        print(f"  - Total messages: {metrics['total_messages']}")
        print(f"  - Outliers: {metrics.get('outliers', 0)}")

        # Get topic info
        info = topic_model.get_topic_info()
        info.insert(0, "message_role", role)
        results.append(info)

        # Extract keywords
        keywords_df = summarise_topics(topic_model, role, args.top_n)
        topic_keywords_frames.append(keywords_df)

        # Save outputs
        info.to_csv(OUTPUT_BASE / f"topic_info_{role}.csv", index=False)

        message_assignments = pd.DataFrame(
            {
                "message_role": role,
                "topic_id": topics,
                "original_message": list(original_texts),
                "cleaned_message": list(cleaned_texts),
            }
        )
        message_assignments.to_csv(OUTPUT_BASE / f"message_topics_{role}.csv", index=False)

        print(f"✓ Saved outputs for {role}")

    # Consolidate outputs
    if topic_keywords_frames:
        pd.concat(topic_keywords_frames, ignore_index=True).to_csv(
            OUTPUT_BASE / "topic_keywords.csv", index=False
        )
    else:
        pd.DataFrame(columns=["message_role", "topic_id", "keywords", "message_count"]).to_csv(
            OUTPUT_BASE / "topic_keywords.csv", index=False
        )

    # Save metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(OUTPUT_BASE / "topic_metrics.csv", index=False)
        print("\n✓ Topic quality metrics saved to topic_metrics.csv")

    print(f"\n{'='*60}")
    print("✓ Topic modeling completed successfully!")
    print(f"✓ Outputs written to {OUTPUT_BASE}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
