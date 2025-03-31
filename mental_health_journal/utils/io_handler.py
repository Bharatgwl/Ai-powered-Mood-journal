# import os
# import pandas as pd
# from datetime import datetime
# from transformers import pipeline
# # Constants
# DATA_DIR = "data"
# DATA_PATH = os.path.join(DATA_DIR, "entries.csv")

# # Ensure data folder exists
# os.makedirs(DATA_DIR, exist_ok=True)
# classifier = pipeline("sentiment-analysis")
# def load_entries():
#     """Load journal entries from CSV."""
#     if os.path.exists(DATA_PATH):
#         df = pd.read_csv(DATA_PATH, quotechar='"')
#         if "timestamp" in df.columns:
#             df["timestamp"] = pd.to_datetime(df["timestamp"])
#         return df
#     else:
#         return pd.DataFrame(columns=["timestamp", "entry", "mood"])

# def save_entry(entry_text):
#     """Save a new journal entry with mood detection."""
#     df = load_entries()

#     mood = detect_mood(entry_text)
#     new_entry = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "entry": entry_text.strip(),
#         "mood": mood
#     }

#     df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
#     df.to_csv(DATA_PATH, index=False)
#     return mood

# # def detect_mood(text):
# #     """Basic rule-based mood detector."""
# #     text = text.lower()
# #     if any(word in text for word in ["happy", "grateful", "joy", "excited", "amazing", "peaceful", "fun"]):
# #         return "positive"
# #     elif any(word in text for word in ["sad", "anxious", "lonely", "angry", "low", "disappointed", "drained"]):
# #         return "negative"
# #     else:
# #         return "neutral"


# def detect_mood(text):
  
#     result = classifier(text)[0]
#     label = result["label"]
#     if label == "POSITIVE":
#         return "positive"
#     elif label == "NEGATIVE":
#         return "negative"
#     else:
#         return "neutral"


import os
import pandas as pd
from datetime import datetime
from transformers import pipeline

# Constants
DATA_PATH = "data/entries.csv"
os.makedirs("data", exist_ok=True)

# Load BERT sentiment model once
_classifier = pipeline("sentiment-analysis", framework="pt")


def detect_mood(text):
    """Detects mood using BERT sentiment analysis."""
    result = _classifier(text)[0]
    label = result["label"]
    return "positive" if label == "POSITIVE" else "negative" if label == "NEGATIVE" else "neutral"

def load_entries():
    """Loads journal entries from CSV."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, quotechar='"')
        return df
    return pd.DataFrame(columns=["timestamp", "entry", "mood"])

# def save_entry(entry_text):
#     """Saves a new entry with mood detection."""
#     mood = detect_mood(entry_text)
#     df = load_entries()
#     new_entry = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "entry": entry_text.strip(),
#         "mood": mood
#     }
#     df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
#     df.to_csv(DATA_PATH, index=False)
#     return mood
def save_entry(entry_text, mood):
    import pandas as pd
    from datetime import datetime
    import os

    filename = "data/entries.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "entry": entry_text,
        "mood": mood
    }])

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(filename, index=False)
    return mood
