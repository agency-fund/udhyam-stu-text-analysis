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
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from bertopic import BERTopic  # type: ignore
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from umap import UMAP
    from hdbscan import HDBSCAN
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependencies for topic modeling.\n"
        "Install with: pip install bertopic[visualization] sentence-transformers umap-learn hdbscan"
    ) from exc

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


DEFAULT_INPUT = "data/cleaned/messages_translated.csv"
OUTPUT_BASE = Path("data/analysis")

MIN_MESSAGE_TOKENS = 4
SHORT_RESPONSES = {
    "ok",
    "okay",
    "yes",
    "no",
    "yup",
    "nope",
    "sure",
    "done",
    "fine",
    "alright",
    "right",
    "k",
    "kk",
    "hmm",
    "hmmm",
    "hm",
    "h",
    "mhm",
    "mm",
    "mmm",
    "understood",
    "noted",
    "noted sir",
    "yes sir",
    "ok sir",
}

BANNED_LABEL_WORDS = {
    "chat",
    "message",
    "messages",
    "cluster",
    "clusters",
    "theme",
    "themes",
    "topic",
    "topics",
    "conversation",
    "conversations",
}

INTENT_SYNONYM_MAP = {
    "register": "Registration",
    "registration": "Registration",
    "signup": "Registration",
    "log": "Registration",
    "deadline": "Deadlines",
    "deadlines": "Deadlines",
    "submit": "Submission",
    "submission": "Submission",
    "translate": "Translation Help",
    "translation": "Translation Help",
    "meaning": "Translation Help",
    "idea": "Idea Support",
    "ideas": "Idea Support",
    "profile": "Account Support",
    "account": "Account Support",
    "teacher": "Teacher Support",
    "mentor": "Teacher Support",
    "product": "Product Support",
    "materials": "Product Support",
    "purpose": "Program Purpose",
    "kb": "Program Purpose",
}

GUIDED_TOPIC_SEEDS = [
    ["register", "team", "portal"],
    ["program", "deadline", "submit"],
    ["business", "idea", "help"],
    ["product", "make", "material"],
    ["translation", "meaning", "english"],
    ["teacher", "support", "mentor"],
    ["purpose", "kb", "use"],
    ["task", "today", "next"],
    ["account", "profile", "login"],
    ["eligibility", "rules", "program"],
]

ASSISTANT_PREFIX_RE = re.compile(r"^(sure|please provide|please share|provide|send|done|okay|ok|thank|thanks)\b", re.IGNORECASE)
ASSISTANT_TEMPLATE_PATTERNS = [
    re.compile(r"please provide (the )?text", re.IGNORECASE),
    re.compile(r"summari[sz]e this", re.IGNORECASE),
    re.compile(r"upload (your )?idea", re.IGNORECASE),
    re.compile(r"send (it|the details)", re.IGNORECASE),
    re.compile(r"share (the )?(details|text)", re.IGNORECASE),
]

BANNED_ASSISTANT_LABEL_TOKENS = {
    "sure",
    "please",
    "provide",
    "help request",
    "label accordingly",
    "assistant",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Translated CSV input")
    parser.add_argument("--min-topic-size", type=int, default=15, help="Min messages per topic (controls cluster granularity)")
    parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min cluster size")
    parser.add_argument("--top-n", type=int, default=10, help="Top keywords per topic")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist parameter")
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon to merge near-duplicate topics",
    )
    parser.add_argument("--reduce-topics", action="store_true", help="Auto-reduce topics after fitting")
    parser.add_argument(
        "--no-reduce-topics",
        action="store_false",
        dest="reduce_topics",
        help="Disable automatic topic reduction",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit per role for debugging",
    )
    parser.add_argument(
        "--use-openai-labels",
        action="store_true",
        help="Use OpenAI chat models for post-hoc topic labeling",
    )
    parser.add_argument(
        "--disable-openai-labels",
        action="store_false",
        dest="use_openai_labels",
        help="Disable OpenAI chat models for post-hoc topic labeling",
    )
    parser.add_argument(
        "--use-openai-representation",
        action="store_true",
        dest="use_openai_labels",
        help="Deprecated alias for --use-openai-labels",
    )
    parser.add_argument(
        "--disable-openai-representation",
        action="store_false",
        dest="use_openai_labels",
        help="Deprecated alias for --disable-openai-labels",
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_TOPIC_MODEL", "gpt-4o-mini"),
        help="OpenAI model name for topic representations",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.05,
        help="Sampling temperature for topic labeling prompts",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-mpnet-base-v2",
        help="SentenceTransformer embedding model for document embeddings",
    )
    parser.add_argument(
        "--similarity-merge-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for automatic topic merging",
    )
    parser.add_argument(
        "--max-similarity-merges",
        type=int,
        default=2,
        help="Maximum number of automatic topic merges per role",
    )
    parser.add_argument(
        "--min-message-tokens",
        type=int,
        default=MIN_MESSAGE_TOKENS,
        help="Minimum number of tokens required in a cleaned message",
    )
    parser.add_argument(
        "--seed-topics",
        action="store_true",
        help="Enable seeded topic guidance using predefined intent keywords",
    )
    parser.add_argument(
        "--no-seed-topics",
        action="store_false",
        dest="seed_topics",
        help="Disable seeded topic guidance",
    )
    parser.add_argument(
        "--process-assistant",
        action="store_true",
        help="Also build topics for assistant responses (kept separate)",
    )
    parser.add_argument(
        "--no-process-assistant",
        action="store_false",
        dest="process_assistant",
        help="Skip assistant response modeling",
    )
    parser.set_defaults(use_openai_labels=True, reduce_topics=False, seed_topics=True, process_assistant=True)
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
        "fine",
        "sir",
        "madam",
        "ma'am",
        "maa'm",
        "pls",
        "plz",
        "dear",
        "regards",
        "understood",
        "noted",
        "done",
        "please sir",
        "please madam",
        "teacher",
        "respected",
        "nice",
        "great",
        "thankyou",
        "thankyou sir",
        "super",
        "awesome",
        "chat",
        "message",
        "messages",
        "cluster",
        "clusters",
        "theme",
        "themes",
        "topic",
        "topics",
        "conversation",
        "conversations",
        "sure",
        "please provide",
        "provide",
        "provide the text",
        "provide text",
        "upload",
        "upload your idea",
        "summarize",
        "summarize this",
        "send",
        "send it",
        "done",
        "thanks in advance",
        "please share",
        "kindly",
        "ok",
        "okay",
    }

    platform_noise.update(BANNED_LABEL_WORDS)

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
        "ya",
        "yo",
    }

    return list(english | platform_noise | fillers)


def collapse_repeated_tokens(text: str) -> str:
    """Collapse immediate repeated tokens ("ok ok ok" -> "ok")."""
    tokens = text.split()
    if not tokens:
        return text
    collapsed = [tokens[0]]
    for token in tokens[1:]:
        if token != collapsed[-1]:
            collapsed.append(token)
    return " ".join(collapsed)


def is_assistant_like_message(text: str) -> bool:
    if not text:
        return False
    if ASSISTANT_PREFIX_RE.search(text):
        return True
    for pattern in ASSISTANT_TEMPLATE_PATTERNS:
        if pattern.search(text):
            return True
    return False


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

    # Collapse duplicate tokens
    text = collapse_repeated_tokens(text)

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
        r'\bteam registration\b': 'team_registration',
        r'\bteam register\b': 'team_registration',
        r'\bprogram deadline\b': 'program_deadline',
        r'\bsubmission deadline\b': 'submission_deadline',
        r'\bbusiness idea\b': 'business_idea',
        r'\bproject idea\b': 'project_idea',
        r'\bmake product\b': 'make_product',
        r'\bbuild product\b': 'make_product',
        r'\bmeaning of\b': 'meaning_of',
        r'\btranslate into\b': 'translate_into',
        r'\bteacher support\b': 'teacher_support',
        r'\bmentor support\b': 'teacher_support',
        r'\bbb program\b': 'bb_program',
        r'\bkb purpose\b': 'kb_purpose',
        r'\bwhat next\b': 'what_next',
        r'\baccount issue\b': 'account_issue',
        r'\blogin issue\b': 'login_issue',
    }

    for pattern, replacement in phrase_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


def clean_message(text: str) -> str:
    """Apply full preprocessing pipeline to a message."""
    text = preprocess_text(text)
    text = merge_common_phrases(text)
    text = lemmatize_text(text)
    if is_assistant_like_message(text):
        return ""
    return text


def create_topic_model(
    stop_words: list[str],
    min_topic_size: int,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    min_cluster_size: int = 15,
    embedding_model_name: str = "all-mpnet-base-v2",
    cluster_selection_epsilon: float = 0.0,
    use_seed_topics: bool = True,
) -> BERTopic:
    """Create improved BERTopic model with better embeddings and tuned parameters.

    Args:
        stop_words: List of stopwords to exclude.
        min_topic_size: Minimum messages required per topic.
        n_neighbors: UMAP n_neighbors (lower for denser clusters).
        min_dist: UMAP min_dist (lower = tighter clusters).
        min_cluster_size: HDBSCAN minimum cluster size.
        embedding_model_name: SentenceTransformer model name.
        cluster_selection_epsilon: HDBSCAN epsilon to merge nearby clusters.

    Returns:
        Configured BERTopic model.
    """
    embedding_model = SentenceTransformer(embedding_model_name)

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
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass (better for varied densities)
        cluster_selection_epsilon=cluster_selection_epsilon or 0.0,
        prediction_data=True,
    )

    # Configure vectorizer for keyword extraction
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),  # Include bigrams for short message context
        min_df=3,
        max_df=0.65,
    )

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        seed_topic_list=GUIDED_TOPIC_SEEDS if use_seed_topics else None,
        calculate_probabilities=False,  # Faster computation
        verbose=True,  # Show progress
    )


def build_openai_client(enable: bool, model_name: str, temperature: float) -> Optional[dict]:
    """Initialise OpenAI client for post-hoc topic labelling."""

    if not enable:
        return None

    if openai is None:
        raise SystemExit(
            "OpenAI topic labelling requires the 'openai' package. Install with: pip install 'openai>=1.0.0'."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set to use OpenAI topic labelling.")

    client = openai.OpenAI(api_key=api_key)
    return {
        "client": client,
        "model": model_name,
        "temperature": temperature,
    }


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

    active_topics = {topic for topic in (topic_model.topics_ or []) if topic != -1}
    if active_topics:
        topics_info = topics_info[topics_info.Topic.isin(active_topics)]

    if topics_info.empty:
        return {
            "num_topics": 0,
            "diversity": 0.0,
            "coherence_c_npmi": float("nan"),
            "avg_topic_size": 0,
            "median_topic_size": 0,
            "total_messages": len(texts),
            "outliers": 0,
        }

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

    try:
        tokenised_texts = [doc.split() for doc in texts if doc.strip()]
        topic_word_lists = [
            [word for word, _ in topic_model.get_topic(topic_id)]
            for topic_id in topics_info.Topic.tolist()
        ]
        coherence = topic_model.get_coherence(
            topics=topic_word_lists,
            texts=tokenised_texts,
            measure="c_npmi",
        )
    except Exception:  # pragma: no cover - optional metric may fail if dependencies missing
        coherence = float("nan")

    metrics = {
        "num_topics": len(topics_info),
        "diversity": round(diversity, 3),
        "coherence_c_npmi": coherence if isinstance(coherence, (float, int)) else float("nan"),
        "avg_topic_size": int(topics_info.Count.mean()),
        "median_topic_size": int(topics_info.Count.median()),
        "total_messages": len(texts),
        "outliers": int(topics_info[topics_info.Topic == -1].Count.sum()) if -1 in topics_info.Topic.values else 0,
    }

    return metrics


def _topic_label_representation(topic_model: BERTopic, topic_id: int) -> str:
    terms = topic_model.get_topic(topic_id) or []
    keywords = " ".join(word for word, _ in terms[:10])
    representative_docs = topic_model.get_representative_docs(topic_id) or []
    doc_sample = " ".join(representative_docs[:2])
    return (keywords + " " + doc_sample).strip()


def merge_similar_topics(
    topic_model: BERTopic,
    documents: list[str],
    similarity_threshold: float,
    max_merges: int,
) -> int:
    """Merge topics with cosine similarity above the threshold."""

    topics_dict = topic_model.get_topics()
    if not topics_dict:
        return 0

    encoder = topic_model.embedding_model
    if not hasattr(encoder, "encode"):
        return 0

    topic_ids = [topic_id for topic_id in topics_dict.keys() if topic_id != -1]
    if len(topic_ids) < 2:
        return 0

    merges_performed = 0

    while merges_performed < max_merges and len(topic_ids) >= 2:
        representations = [_topic_label_representation(topic_model, tid) for tid in topic_ids]
        embeddings = np.asarray(encoder.encode(representations, show_progress_bar=False))
        if embeddings.size == 0:
            break

        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0.0)
        max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        max_similarity = similarity_matrix[max_idx]

        if max_similarity < similarity_threshold:
            break

        topic_a, topic_b = topic_ids[max_idx[0]], topic_ids[max_idx[1]]

        try:
            topic_model.merge_topics(documents, topics_to_merge=[topic_a, topic_b])
        except ValueError:  # pragma: no cover - stale topic IDs
            break

        merges_performed += 1
        topics_dict = topic_model.get_topics()
        topic_ids = [topic_id for topic_id in topics_dict.keys() if topic_id != -1]

    return merges_performed


def filter_trivial_topics(topic_model: BERTopic, documents: list[str], min_tokens: int) -> None:
    """Mark topics containing only trivial short messages as outliers."""

    assignments = getattr(topic_model, "topics_", None)
    if assignments is None:
        return

    filtered_assignments = []
    for topic_id, message in zip(assignments, documents):
        if topic_id == -1:
            filtered_assignments.append(topic_id)
            continue

        token_count = len(message.split())
        if token_count < min_tokens or message in SHORT_RESPONSES:
            filtered_assignments.append(-1)
        else:
            filtered_assignments.append(topic_id)

    topic_model.topics_ = filtered_assignments


def drop_low_quality_topics(topic_model: BERTopic, documents: list[str]) -> None:
    assignments = getattr(topic_model, "topics_", None)
    if assignments is None:
        return

    updated_assignments = []
    by_topic: dict[int, list[str]] = {}
    for topic_id, message in zip(assignments, documents):
        by_topic.setdefault(topic_id, []).append(message)

    drop_ids: set[int] = set()
    for topic_id, msgs in by_topic.items():
        if topic_id == -1:
            continue
        lengths = [len(msg.split()) for msg in msgs if msg.strip()]
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        assistant_like = sum(1 for msg in msgs if is_assistant_like_message(msg))
        assistant_ratio = assistant_like / len(msgs) if msgs else 0.0
        if avg_len < 3 or assistant_ratio > 0.6:
            drop_ids.add(topic_id)

    if not drop_ids:
        return

    for topic_id, message in zip(assignments, documents):
        if topic_id in drop_ids:
            updated_assignments.append(-1)
        else:
            updated_assignments.append(topic_id)

    topic_model.topics_ = updated_assignments


def _tokenize_label_text(text: str) -> list[str]:
    tokens = re.split(r"[\s_/,-]+", text.lower())
    return [token for token in tokens if token and token not in BANNED_LABEL_WORDS]


def _apply_synonyms(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        mapped = INTENT_SYNONYM_MAP.get(token, token)
        expanded.extend(mapped.lower().split())
    return expanded


def _generate_label_from_keywords(topic_words: list[tuple[str, float]]) -> list[str]:
    """Generate a human-readable label from topic keywords.

    Prioritizes action words and noun phrases to create meaningful labels.
    """
    keywords: list[str] = []

    # Extract top keywords, prioritizing those with underscores (compound terms)
    compound_terms = []
    simple_terms = []

    for word, score in topic_words[:8]:  # Look at more keywords for better context
        if '_' in word:
            compound_terms.append(word)
        else:
            simple_terms.append(word)

    # Process compound terms first (they're usually more descriptive)
    for term in compound_terms[:3]:
        keywords.extend(_tokenize_label_text(term))

    # Add simple terms if we don't have enough
    for term in simple_terms[:5]:
        keywords.extend(_tokenize_label_text(term))

    if not keywords and topic_words:
        keywords = [topic_words[0][0]]

    keywords = _apply_synonyms(keywords)

    # Preserve order while removing duplicates
    seen = set()
    deduped = []
    for token in keywords:
        if token not in seen:
            seen.add(token)
            deduped.append(token)

    # Ensure we have at least 2-3 words for context
    if len(deduped) < 2 and topic_words:
        # Add more tokens from top keywords
        for word, _ in topic_words[:10]:
            tokens = _tokenize_label_text(word)
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    deduped.append(token)
                    if len(deduped) >= 3:
                        break
            if len(deduped) >= 3:
                break

    return deduped[:5]


def _format_label(tokens: list[str]) -> str:
    cleaned = [token for token in tokens if token and token not in BANNED_LABEL_WORDS]
    if len(cleaned) < 2 and tokens:
        cleaned = tokens[:2]
    cleaned = cleaned[:5]
    return " ".join(word.capitalize() for word in cleaned)


def generate_llm_topic_labels(
    topic_model: BERTopic,
    cleaned_texts: list[str],
    bundle: Optional[dict],
    role: str,
) -> Dict[int, str]:
    if not bundle:
        return {}

    client = bundle["client"]
    model = bundle["model"]
    temperature = bundle["temperature"]
    topics_info = topic_model.get_topic_info()
    topic_ids = [row.Topic for _, row in topics_info.iterrows() if row.Topic != -1]
    label_map: Dict[int, str] = {}
    banned = ", ".join(sorted(BANNED_LABEL_WORDS | BANNED_ASSISTANT_LABEL_TOKENS))

    assignments = topic_model.topics_ or []

    for topic_id in topic_ids:
        # Get more representative documents for better generalization
        representative_docs = topic_model.get_representative_docs(topic_id) or []
        if not representative_docs:
            sample_docs = [
                message
                for message, assignment in zip(cleaned_texts, assignments)
                if assignment == topic_id
            ][:10]  # Increased from 5 to 10 for better coverage
        else:
            sample_docs = representative_docs[:10]

        if not sample_docs:
            continue

        # Get topic keywords for context
        topic_words = topic_model.get_topic(topic_id) or []
        top_keywords = ", ".join(word for word, _ in topic_words[:8])

        bullet_list = "\n".join(f"- {doc}" for doc in sample_docs)
        prompt = (
            "You are analyzing topics from a chatbot conversation.\n"
            "Your task: Create a general intent label that captures the COMMON THEME across ALL the messages below.\n\n"
            "Important guidelines:\n"
            "- Identify what ALL messages have in common, not just one specific example\n"
            "- Use 2-5 words in Title Case (e.g., 'Product Creation Help', 'Team Registration Support')\n"
            "- Focus on the general action or need, not specific details\n"
            "- The label should apply to most/all of the messages shown\n"
            f"- Avoid these generic words: {banned}\n"
            "- No punctuation, no explanations\n\n"
            f"Top keywords for this topic: {top_keywords}\n\n"
            f"Messages:\n{bullet_list}\n\n"
            "General intent label:"
        )

        label_text = ""
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=20,
            )
            label_text = getattr(response, "output_text", "").strip()
        except Exception:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You label user help requests with concise intent phrases.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=20,
                )
                label_text = response.choices[0].message.content.strip()
            except Exception:
                label_text = ""

        if label_text:
            # Keep just first line and strip punctuation
            label_text = label_text.splitlines()[0].strip().strip("-:")

            # Validate that the label is general enough and relates to keywords
            # If label seems too specific (mentions very specific items), flag it
            specific_indicators = [
                'colgate', 'neem', 'orange', 'mat', 'pistachio', 'dairy', 'sock',
                'instagram', 'facebook', 'whatsapp'  # Specific products/platforms
            ]
            if any(indicator in label_text.lower() for indicator in specific_indicators):
                # Label is too specific, generate from keywords instead
                print(f"  ⚠ Topic {topic_id}: Label '{label_text}' too specific, using keyword-based fallback")
                label_text = ""

            if label_text:
                label_map[topic_id] = label_text

    return label_map


def apply_intent_labels(topic_model: BERTopic, initial_labels: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """Sanitize or regenerate topic labels to enforce intent-style naming."""

    topics_info = topic_model.get_topic_info()
    label_map: Dict[int, str] = {}
    active_topics = {topic for topic in (topic_model.topics_ or []) if topic != -1}

    for _, row in topics_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1 or (active_topics and topic_id not in active_topics):
            continue

        raw_label = ""
        if initial_labels and topic_id in initial_labels:
            raw_label = initial_labels[topic_id]
        else:
            raw_label = str(row.get("Name", ""))

        raw_label_lower = raw_label.lower()
        tokens = _tokenize_label_text(raw_label) if raw_label else []
        tokens = _apply_synonyms(tokens)

        topic_words = topic_model.get_topic(topic_id) or []

        banned_strings = BANNED_LABEL_WORDS | BANNED_ASSISTANT_LABEL_TOKENS
        banned_hit = any(banned in raw_label_lower for banned in banned_strings)
        if (not tokens) or banned_hit or len(tokens) < 2:
            tokens = _generate_label_from_keywords(topic_words)

        if len(tokens) < 2:
            additional = _generate_label_from_keywords(topic_words)
            tokens = tokens + [tok for tok in additional if tok not in tokens]

        formatted = _format_label(tokens)
        formatted_lower = formatted.lower()
        if any(banned in formatted_lower for banned in banned_strings):
            tokens = _generate_label_from_keywords(topic_words)
            formatted = _format_label(tokens)

        label_map[topic_id] = formatted

    if label_map:
        topic_model.set_topic_labels(label_map)

    return label_map


def topic_diagnostics(
    topic_model: BERTopic,
    cleaned_texts: list[str],
    label_map: Dict[int, str],
    role: str,
    original_texts: list[str],
) -> pd.DataFrame:
    assignments_raw = topic_model.topics_ or []
    assignments = list(assignments_raw)
    df_rows = []
    topic_ids = sorted({topic_id for topic_id in assignments if topic_id != -1})

    for topic_id in topic_ids:
        topic_words = topic_model.get_topic(topic_id) or []
        keywords = [word for word, _ in topic_words[:10]]

        topic_messages = [
            message for message, assignment in zip(cleaned_texts, assignments) if assignment == topic_id
        ]
        topic_originals = [
            message for message, assignment in zip(original_texts, assignments) if assignment == topic_id
        ]
        avg_length = float(np.mean([len(msg.split()) for msg in topic_messages])) if topic_messages else 0.0

        print("-" * 40)
        print(f"Role: {role} | Topic {topic_id} -> {label_map.get(topic_id, f'Topic {topic_id}')}")
        print(f"  Keywords: {', '.join(keywords)}")
        print(f"  Message count: {len(topic_messages)}")
        print(f"  Avg message length: {avg_length:.2f} tokens")
        if avg_length < 4:
            print("  ⚠ Average message length below 4 tokens - review needed")
        print("  Top user messages:")
        if topic_originals:
            for msg in topic_originals[:5]:
                print(f"    • {msg}")
        else:
            for msg in topic_messages[:5]:
                print(f"    • {msg}")

        examples = topic_originals[:3] or topic_messages[:3]
        while len(examples) < 3:
            examples.append("")

        df_rows.append(
            {
                "intent_label": label_map.get(topic_id, f"Topic {topic_id}"),
                "keywords": ", ".join(keywords),
                "message_count": len(topic_messages),
                "example_1": examples[0],
                "example_2": examples[1],
                "example_3": examples[2],
            }
        )

    return pd.DataFrame(
        df_rows,
        columns=["intent_label", "keywords", "message_count", "example_1", "example_2", "example_3"],
    )


def summarise_topics(topic_model: BERTopic, role: str, top_n: int, label_map: Dict[int, str]) -> pd.DataFrame:
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
    info.rename(columns={"Name": "intent_label", "Count": "message_count", "Topic": "topic_id"}, inplace=True)

    rows = []
    for topic_id in info["topic_id"].tolist():
        terms = topic_model.get_topic(topic_id)
        if not terms:
            continue
        keywords = ", ".join(word for word, _ in terms[:top_n])
        intent_label = label_map.get(topic_id, info.loc[info.topic_id == topic_id, "intent_label"].iloc[0])
        rows.append({
            "message_role": role,
            "topic_id": topic_id,
            "intent_label": intent_label,
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

    openai_bundle = build_openai_client(
        args.use_openai_labels,
        args.openai_model,
        args.openai_temperature,
    )

    role_columns = {
        "user": "user_msg_en",
    }
    if args.process_assistant:
        role_columns["assistant"] = "ai_msg_en"

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

        # Filter out empty, trivial, or too-short messages after cleaning
        cleaned_pairs = [
            (original, cleaned)
            for original, cleaned in zip(texts.tolist(), cleaned_texts)
            if cleaned.strip()
            and cleaned not in SHORT_RESPONSES
            and not is_assistant_like_message(cleaned)
            and len(cleaned.split()) >= args.min_message_tokens
        ]
        if not cleaned_pairs:
            print(f"⚠ No valid messages after preprocessing for {role}, skipping...")
            continue

        original_texts, cleaned_texts = zip(*cleaned_pairs)
        original_texts = list(original_texts)
        cleaned_texts = list(cleaned_texts)
        print(f"✓ {len(cleaned_texts)} messages after preprocessing")

        # Create and fit model
        print("Creating BERTopic model with improved parameters...")
        topic_model = create_topic_model(
            stop_words,
            args.min_topic_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            min_cluster_size=args.min_cluster_size,
            embedding_model_name=args.embedding_model,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            use_seed_topics=args.seed_topics,
        )

        print("Fitting topics...")
        topics, _ = topic_model.fit_transform(list(cleaned_texts))

        # Get initial topic count
        initial_topics = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"✓ Initial topics found: {initial_topics}")

        metrics_before = compute_topic_metrics(topic_model, list(cleaned_texts))
        metrics_before["stage"] = "initial"
        metrics_before["message_role"] = role
        all_metrics.append(metrics_before)

        print(f"  ↳ Initial diversity: {metrics_before['diversity']:.3f}")
        print(f"  ↳ Initial coherence (c_npmi): {metrics_before['coherence_c_npmi']}")

        # Apply topic reduction if requested
        if args.reduce_topics and initial_topics > 0:
            print("Applying automatic topic reduction...")
            topic_model.reduce_topics(list(cleaned_texts), nr_topics="auto")
            topics = topic_model.topics_
            final_topics = len(set(topics)) - (1 if -1 in topics else 0)
            print(f"✓ Topics after reduction: {final_topics} (reduced by {initial_topics - final_topics})")

        print("Merging highly similar topics...")
        merges = merge_similar_topics(
            topic_model,
            list(cleaned_texts),
            similarity_threshold=args.similarity_merge_threshold,
            max_merges=args.max_similarity_merges,
        )
        if merges:
            print(f"✓ Merged {merges} pairs of similar topics")
        else:
            print("✓ No similar topics exceeded merge threshold")

        print("Dropping trivial short-message topics...")
        filter_trivial_topics(topic_model, list(cleaned_texts), args.min_message_tokens)
        drop_low_quality_topics(topic_model, list(cleaned_texts))
        topics = list(topic_model.topics_ or [])

        print("Applying intent-style topic labels and diagnostics...")
        llm_label_map = generate_llm_topic_labels(topic_model, list(cleaned_texts), openai_bundle, role)
        label_map = apply_intent_labels(topic_model, llm_label_map)
        diagnostics_df = topic_diagnostics(
            topic_model,
            list(cleaned_texts),
            label_map,
            role,
            list(original_texts),
        )
        diagnostics_df.to_csv(OUTPUT_BASE / f"topic_examples_{role}.csv", index=False)

        # Compute metrics
        print("Computing topic quality metrics...")
        metrics = compute_topic_metrics(topic_model, list(cleaned_texts))
        metrics["stage"] = "post_processing"
        metrics["message_role"] = role
        all_metrics.append(metrics)

        final_topic_count = metrics["num_topics"]
        if final_topic_count < 10:
            print("⚠ Warning: fewer than 10 topics detected. Consider adjusting clustering parameters.")
        elif final_topic_count > 30:
            print("⚠ Warning: more than 30 topics detected. Review for over-fragmentation.")
        if metrics["diversity"] < 0.6:
            print("⚠ Topic diversity below target of 0.60. Investigate stopwords or min_df settings.")

        print(f"\nTopic Quality Metrics for {role} (post-processing):")
        print(f"  - Number of topics: {metrics['num_topics']}")
        print(f"  - Topic diversity: {metrics['diversity']:.3f}")
        print(f"  - Topic coherence (c_npmi): {metrics['coherence_c_npmi']}")
        print(f"  - Avg topic size: {metrics['avg_topic_size']}")
        print(f"  - Median topic size: {metrics['median_topic_size']}")
        print(f"  - Total messages: {metrics['total_messages']}")
        print(f"  - Outliers: {metrics.get('outliers', 0)}")

        # Get topic info with intent labels and representative keywords
        info = topic_model.get_topic_info()
        info = info[info.Topic != -1].copy()
        info["intent_label"] = info["Topic"].map(label_map).fillna(info["Name"])
        info["top_keywords"] = info["Topic"].apply(
            lambda tid: ", ".join(word for word, _ in (topic_model.get_topic(tid) or [])[: args.top_n])
        )
        info.insert(0, "message_role", role)
        results.append(info)

        # Extract keywords
        keywords_df = summarise_topics(topic_model, role, args.top_n, label_map)
        topic_keywords_frames.append(keywords_df)

        # Save outputs
        info.to_csv(OUTPUT_BASE / f"topic_info_{role}.csv", index=False)

        message_assignments = pd.DataFrame(
            {
                "message_role": role,
                "topic_id": topics,
                "intent_label": [label_map.get(topic_id, f"Topic {topic_id}") if topic_id != -1 else "Outlier" for topic_id in topics],
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
        pd.DataFrame(columns=["message_role", "topic_id", "intent_label", "keywords", "message_count"]).to_csv(
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
