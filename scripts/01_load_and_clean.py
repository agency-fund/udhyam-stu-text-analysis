import pandas as pd
from datetime import datetime
from googletrans import Translator
import time
import re

# Load the CSV file
csv_path = '/Users/michaelfive/Library/CloudStorage/GoogleDrive-michael@agency.fund/My Drive/Work/Udhyam/Data/AI Help (Udhyam) student usage_Table - Sheet1.csv'
messages = pd.read_csv(csv_path)

print("Original data shape:", messages.shape)
print("\nFirst few rows:")
print(messages.head())
print("\nColumn names:")
print(messages.columns.tolist())
print("\nData types:")
print(messages.dtypes)
print("\nSample session_id values:")
print(messages['session_id'].head(10))

# Parse session_id into datetime and conversation_id
def parse_session_id(session_id):
    """
    Parse session_id string into datetime and conversation_id.
    Expected format: "YYYY-MM-DD HH:MM:SS<conversation_id>"
    """
    session_id_str = str(session_id)

    # The datetime part is the first 19 characters: "YYYY-MM-DD HH:MM:SS"
    datetime_str = session_id_str[:19]

    # The rest is the conversation_id
    conversation_id = session_id_str[19:]

    # Convert datetime string to datetime object
    dt = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S')

    return dt, conversation_id

# Apply the parsing function
messages[['datetime', 'conversation_id']] = messages['session_id'].apply(
    lambda x: pd.Series(parse_session_id(x))
)

# Remove session_id column
messages = messages.drop(columns=['session_id'])

# Rename columns
messages = messages.rename(columns={
    'Question': 'user_msg',
    'Category': 'user_msg_category',
    'Response': 'ai_msg',
    'translated_answer': 'ai_msg_en'
})

# Remove cal_role column
messages = messages.drop(columns=['cal_role'])

# Reorder columns to put datetime and conversation_id first
columns = ['datetime', 'conversation_id'] + [col for col in messages.columns if col not in ['datetime', 'conversation_id']]
messages = messages[columns]

print("\n" + "="*80)
print("AFTER BASIC CLEANING:")
print("="*80)
print("\nData shape:", messages.shape)
print("\nColumn names:")
print(messages.columns.tolist())

# ============================================================================
# TRANSLATE USER MESSAGES TO ENGLISH
# ============================================================================

print("\n" + "="*80)
print("TRANSLATING USER MESSAGES TO ENGLISH...")
print("="*80)

translator = Translator()

def is_likely_english(text):
    """Quick check if text is likely in English already"""
    if pd.isna(text) or text == '':
        return True

    text_str = str(text).lower()

    # If it's very short (<=3 chars), consider it English
    if len(text_str) <= 3:
        return True

    # Check for non-Latin scripts (Devanagari for Hindi, Gurmukhi for Punjabi)
    if re.search(r'[\u0900-\u097F\u0A00-\u0A7F]', text_str):
        return False

    # Check for common English words
    common_english_words = {'the', 'is', 'are', 'was', 'were', 'ok', 'yes', 'no',
                           'hello', 'hi', 'thank', 'please', 'can', 'will', 'how',
                           'what', 'when', 'where', 'who', 'why', 'this', 'that'}

    words = text_str.split()
    if any(word in common_english_words for word in words):
        return True

    # If it contains mostly Latin characters, assume English
    latin_chars = sum(1 for c in text_str if ord(c) < 128)
    if len(text_str) > 0 and latin_chars / len(text_str) > 0.8:
        return True

    return False

def translate_to_english(text):
    """Translate text to English with error handling"""
    if pd.isna(text) or text == '':
        return ''

    if is_likely_english(text):
        return str(text)

    try:
        result = translator.translate(str(text), dest='en')
        time.sleep(0.1)
        return result.text
    except Exception as e:
        print(f"\nError translating: {text[:50]}... | Error: {e}")
        return str(text)

# Filter messages that need translation
needs_translation = []
already_english = []

for i, text in enumerate(messages['user_msg']):
    if is_likely_english(text):
        already_english.append(i)
    else:
        needs_translation.append(i)

print(f"Messages already in English: {len(already_english)}")
print(f"Messages needing translation: {len(needs_translation)}")

# Translate messages
user_msg_en_list = [None] * len(messages)

# Copy English messages as-is
for idx in already_english:
    user_msg_en_list[idx] = str(messages.iloc[idx]['user_msg'])

# Translate non-English messages
print("Translating non-English messages...")
for i, idx in enumerate(needs_translation):
    text = messages.iloc[idx]['user_msg']
    translated = translate_to_english(text)
    user_msg_en_list[idx] = translated

    if (i + 1) % 50 == 0:
        print(f"  Translated {i + 1}/{len(needs_translation)} messages")
        time.sleep(2)

# Add translated column next to user_msg
messages.insert(messages.columns.get_loc('user_msg') + 1, 'user_msg_en', user_msg_en_list)

print("\n" + "="*80)
print("FINAL CLEANED DATA:")
print("="*80)
print("\nData shape:", messages.shape)
print("\nColumn names:")
print(messages.columns.tolist())
print("\nFirst few rows:")
print(messages.head(10))
print("\nData types:")
print(messages.dtypes)

# Save final translated data
output_path = 'data/messages_translated.csv'
messages.to_csv(output_path, index=False)
print(f"\nFinal data saved to: {output_path}")
