# Mood Journal AI

A Flask-based AI-powered mood journaling app that analyzes user sentiments using NLP models. The app provides insights into emotional patterns through visualization.

## Features
- Sentiment analysis using Hugging Face's `sentiment-analysis` pipeline (PyTorch-based)
- User-friendly frontend built with **Streamlit**
- Stores user moods in **SQLite** for trend tracking
- Data visualization for mood trends

## Tech Stack
- **Backend:** Flask, Python
- **Frontend:** Streamlit
- **NLP Model:** Hugging Face Transformers (PyTorch-based)
- **Database:** SQLite
- **Libraries:** NLTK, TextBlob, Matplotlib, Seaborn

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mood-journal-ai.git
cd mood-journal-ai
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## Future Enhancements
- Improve the model's accuracy using fine-tuned transformers
- Add authentication for personalized journaling
- Implement cloud database support for scalability

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

## License
MIT License