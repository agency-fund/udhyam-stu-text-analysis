# Udhyam Student Text Analysis

Data cleaning and translation pipeline for Udhyam AI chatbot student interactions.

## Repository Structure

```
.
├── data/
│   ├── messages_cleaned.csv                      # Cleaned dataset
│   └── messages_translated_openai.csv            # Final translated dataset (after script completes)
├── scripts/
│   ├── 01_load_and_clean.py                      # Data loading and cleaning
│   └── 02_translate_messages.py                  # OpenAI translation with Hinglish detection
├── notebooks/                                     # Jupyter notebooks for analysis
├── requirements.txt                               # Python dependencies
├── .env.example                                   # Template for API keys
└── README.md
```

## Scripts

### 01_load_and_clean.py
- Loads raw CSV data
- Parses `session_id` into `datetime` and `conversation_id`
- Renames and reorders columns
- Removes unnecessary columns (`session_id`, `cal_role`)

### 02_translate_messages.py
- Translates `user_msg` column to English using OpenAI gpt-4o-mini
- Detects Hinglish (Romanized Hindi/Punjabi) messages
- Handles mixed language text (English, Hindi, Punjabi, Hinglish)
- Real-time cost tracking
- Expected cost: ~$0.23 for full dataset

## Data Schema

**Final columns:**
- `datetime` - Timestamp of message
- `conversation_id` - Conversation identifier
- `user_msg` - Original user message (mixed languages)
- `user_msg_en` - English translation
- `user_msg_category` - Message category
- `ai_msg` - AI response
- `ai_msg_en` - AI response in English
- `cal_state` - Conversation state
- `cal_feedback` - User feedback

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Run scripts in order:
```bash
python scripts/01_load_and_clean.py
python scripts/02_translate_messages.py
```

## Translation Details

The translation script uses OpenAI's gpt-4o-mini model with:
- Temperature: 0.3 (for consistency)
- Specialized system prompt for Indian languages
- Hinglish keyword detection (40+ common words)
- Pre-filtering to skip already-English messages
- Cost tracking and reporting
