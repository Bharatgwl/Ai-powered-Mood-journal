# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# from utils.io_handler import load_entries, save_entry

# # -----------------------
# # Sidebar Navigation
# # -----------------------
# st.sidebar.title("🧠 Mental Health Journal")
# page = st.sidebar.radio("Go to", ["📓 Journal", "📊 Insights"])

# # -----------------------
# # Load Data
# # -----------------------
# df = load_entries()

# # -----------------------
# # 📓 Journal Page
# # -----------------------
# if page == "📓 Journal":
#     st.title("📝 Daily Journal")
#     st.subheader("Write your thoughts below 👇")

#     entry = st.text_area("Today's Entry", height=200)

#     if st.button("Save Entry"):
#         if entry.strip():
#             mood = save_entry(entry)
#             st.success(f"✅ Entry saved! Mood detected: **{mood}**")
#             df = load_entries()  # Refresh data
#         else:
#             st.warning("⚠️ Please write something before saving.")

#     st.subheader("🗂️ Past Entries")
#     if not df.empty:
#         st.dataframe(df[::-1])  # Show latest first
#     else:
#         st.info("No journal entries yet. Start writing!")

# # -----------------------
# # 📊 Insights Page
# # -----------------------
# elif page == "📊 Insights":
#     st.title("📊 Journal Insights")

#     if df.empty:
#         st.info("No data available. Please add some journal entries.")
#         st.stop()

#     # Preprocess date
#     df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

#     df["day"] = df["timestamp"].dt.date

#     # Mood Distribution
#     st.subheader("😊 Mood Distribution")
#     st.bar_chart(df["mood"].value_counts())

#     # Mood Over Time
#     st.subheader("📅 Mood Over Time")
#     fig, ax = plt.subplots()
#     sns.countplot(data=df, x="day", hue="mood", palette="pastel", ax=ax)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     st.pyplot(fig)

#     # Entry Frequency
#     st.subheader("🕒 Entry Frequency Over Time")
#     entry_count = df["day"].value_counts().sort_index()
#     st.line_chart(entry_count)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import pipeline
from utils.io_handler import load_entries, save_entry

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis" , framework="pt")

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🧠 Mental Health Journal")
page = st.sidebar.radio("Go to", ["📓 Journal", "📊 Insights"])

# -----------------------
# Load Data
# -----------------------
df = load_entries()

# -----------------------
# 📓 Journal Page
# -----------------------
if page == "📓 Journal":
    st.title("📝 Daily Journal")
    st.subheader("Write your thoughts below 👇")

    entry = st.text_area("Today's Entry", height=200)

    if st.button("Save Entry"):
        if entry.strip():
            # Predict mood using Transformer
            result = sentiment_pipeline(entry)[0]
            mood = result['label']
            confidence = result['score']

            # Save entry with mood (adjust save_entry to accept mood if needed)
            save_entry(entry, mood)  # assumes save_entry now accepts mood
            st.success(f"✅ Entry saved! Mood detected: **{mood}** ({confidence:.2f})")
            df = load_entries()  # Refresh data
        else:
            st.warning("⚠️ Please write something before saving.")

    st.subheader("🗂️ Past Entries")
    if not df.empty:
        st.dataframe(df[::-1])  # Show latest first
    else:
        st.info("No journal entries yet. Start writing!")

# -----------------------
# 📊 Insights Page
# -----------------------
elif page == "📊 Insights":
    st.title("📊 Journal Insights")

    if df.empty:
        st.info("No data available. Please add some journal entries.")
        st.stop()

    # Preprocess date
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["day"] = df["timestamp"].dt.date

    # Mood Distribution
    st.subheader("😊 Mood Distribution")
    st.bar_chart(df["mood"].value_counts())

    # Mood Over Time
    st.subheader("📅 Mood Over Time")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="day", hue="mood", palette="pastel", ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Entry Frequency
    st.subheader("🕒 Entry Frequency Over Time")
    entry_count = df["day"].value_counts().sort_index()
    st.line_chart(entry_count)
