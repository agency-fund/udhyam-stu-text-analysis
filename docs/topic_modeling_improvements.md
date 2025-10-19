# Topic Modeling Pipeline Improvements

## Overview

The BERTopic pipeline has been significantly improved to produce clearer, more distinct topics for Udhyam student messages (translated from Hindi, Punjabi, and Hinglish to English).

## Key Improvements

### 1. Enhanced Preprocessing

**Text Cleaning:**
- Removes URLs and email addresses
- Filters out numeric IDs and sequences
- Removes platform-specific noise ("kbc", "portal", "please clarify")
- Normalizes whitespace

**Lemmatization:**
- Reduces words to their base forms (e.g., "learning" → "learn")
- Improves topic coherence by grouping inflected forms

**Phrase Merging:**
- Combines common multi-word phrases into single tokens:
  - "entrepreneurial mindset" → "entrepreneurial_mindset"
  - "business model" → "business_model"
  - "people planet profit" → "people_planet_profit"
  - Many more educational/business terms

### 2. Better Embeddings

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

**Why this model?**
- Optimized for paraphrase detection (good for similar/translated content)
- Multilingual support (handles English with influence from Indian languages)
- Smaller and faster than larger models while maintaining quality
- Better semantic understanding than basic word embeddings

### 3. Tuned UMAP Parameters

**Configuration:**
```python
UMAP(
    n_neighbors=30,      # Balance between local and global structure
    n_components=5,      # Dimensionality reduction
    min_dist=0.05,       # Tighter clusters for better separation
    metric='cosine',     # Optimal for text embeddings
)
```

**Impact:**
- Better topic separation in embedding space
- More distinct, less overlapping clusters

### 4. Tuned HDBSCAN Parameters

**Configuration:**
```python
HDBSCAN(
    min_cluster_size=25,           # Larger clusters = fewer, clearer topics
    min_samples=5,                 # More conservative clustering
    cluster_selection_method='eom' # Better for varied densities
)
```

**Impact:**
- Reduces noise and outliers
- Creates more coherent topic groups
- Fewer but higher-quality topics

### 5. Improved Vectorization

**Configuration:**
```python
CountVectorizer(
    ngram_range=(1, 3),  # Capture up to 3-word phrases
    min_df=3,            # Must appear in at least 3 documents
    max_df=0.8,          # Remove if in >80% of documents
)
```

**Impact:**
- Captures meaningful phrases, not just single words
- Filters out both rare noise and overly common terms

### 6. Automatic Topic Reduction

**Feature:** `--reduce-topics` flag

**What it does:**
- Merges similar/overlapping topics after initial fitting
- Uses `nr_topics="auto"` for intelligent reduction
- Produces fewer, more distinct final topics

### 7. Quality Metrics

**Computed for each role (user/assistant):**

- **Number of topics**: Total distinct topics found
- **Diversity score**: Uniqueness of top words across topics (0-1, higher is better)
- **Average topic size**: Mean messages per topic
- **Median topic size**: Robust measure of topic size
- **Total messages**: Documents processed
- **Outliers**: Messages not assigned to any topic

**Saved to:** `data/analysis/topic_metrics.csv`

## Usage

### Basic Usage

```bash
python scripts/05_topic_modeling.py
```

### With Topic Reduction

```bash
python scripts/05_topic_modeling.py --reduce-topics
```

### Custom Parameters

```bash
python scripts/05_topic_modeling.py \
  --min-topic-size 30 \
  --min-cluster-size 30 \
  --n-neighbors 40 \
  --min-dist 0.03 \
  --top-n 12 \
  --reduce-topics
```

### Parameter Tuning Guide

**If topics are too vague/overlapping:**
- Increase `--min-cluster-size` (e.g., 30-35)
- Decrease `--min-dist` (e.g., 0.03)
- Enable `--reduce-topics`

**If topics are too specific/fragmented:**
- Decrease `--min-cluster-size` (e.g., 15-20)
- Increase `--min-dist` (e.g., 0.1)
- Increase `--n-neighbors` (e.g., 40-50)

**If too many outliers:**
- Decrease `--min-cluster-size`
- Increase `--min-dist`

## Outputs

### Files Generated

1. **`topic_info_{role}.csv`**
   - Full topic information from BERTopic
   - Includes all metadata and statistics

2. **`topic_keywords.csv`**
   - Consolidated topic keywords across roles
   - Columns: message_role, topic_id, keywords, message_count

3. **`message_topics_{role}.csv`**
   - Topic assignments for each message
   - Includes both original and cleaned text
   - Useful for validation and debugging

4. **`topic_metrics.csv`** ⭐ NEW
   - Quality metrics for each role
   - Use to assess topic model performance
   - Iterate parameters if metrics are poor

### Quality Assessment

**Good metrics:**
- Diversity: > 0.7 (topics use distinct vocabularies)
- Avg topic size: 30-100 (reasonable groupings)
- Outliers: < 20% of total messages

**Poor metrics:**
- Diversity: < 0.5 (topics overlap too much)
- Avg topic size: < 10 (too fragmented) or > 200 (too general)
- Outliers: > 30% (clustering too aggressive)

## Iteration Strategy

1. **Run with defaults:**
   ```bash
   python scripts/05_topic_modeling.py --reduce-topics
   ```

2. **Check metrics:**
   ```bash
   cat data/analysis/topic_metrics.csv
   ```

3. **Adjust parameters** based on metrics (see tuning guide above)

4. **Re-run and compare:**
   - Look for improved diversity scores
   - Check if topic keywords make more sense
   - Validate with sample messages in `message_topics_{role}.csv`

5. **Iterate until satisfied** with topic quality

## Dependencies

Ensure these are installed:

```bash
pip install bertopic[visualization] sentence-transformers umap-learn hdbscan nltk
```

The script will auto-download required NLTK data (stopwords, wordnet) on first run.

## Performance Notes

- **First run:** Slower (downloads embedding model ~420MB)
- **Subsequent runs:** Model cached, much faster
- **Processing time:** ~2-5 minutes for typical dataset
- **Memory usage:** ~2-4GB depending on message count

## Example Output

```
============================================================
Processing user messages (847 total)
============================================================
Preprocessing texts (cleaning, lemmatization, phrase merging)...
✓ 823 messages after preprocessing
Creating BERTopic model with improved parameters...
Fitting topics...
✓ Initial topics found: 12
Applying automatic topic reduction...
✓ Topics after reduction: 8 (reduced by 4)
Computing topic quality metrics...

Topic Quality Metrics for user:
  - Number of topics: 8
  - Topic diversity: 0.789
  - Avg topic size: 102
  - Median topic size: 95
  - Total messages: 823
  - Outliers: 47
✓ Saved outputs for user
```

## Next Steps

After running topic modeling:

1. Review topics in `topic_keywords.csv`
2. Check quality metrics in `topic_metrics.csv`
3. Examine sample messages for each topic
4. Update report with insights
5. Iterate parameters if needed

---

**Created:** 2025-10-19
**Script:** `scripts/05_topic_modeling.py`
