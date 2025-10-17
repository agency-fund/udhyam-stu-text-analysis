import pandas as pd
from datetime import datetime

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
# Example format: "2025-10-08 20:13:54919501338066"
# Date: "2025-10-08 20:13:54"
# ID: "919501338066"

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
print("AFTER CLEANING:")
print("="*80)
print("\nData shape:", messages.shape)
print("\nColumn names:")
print(messages.columns.tolist())
print("\nFirst few rows:")
print(messages.head(10))
print("\nData types:")
print(messages.dtypes)
print("\nDatetime column info:")
print(messages['datetime'].describe())

# Save cleaned data
output_path = 'data/messages_cleaned.csv'
messages.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")
