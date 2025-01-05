# Smart Search Tool for Analytics Vidhya Free Courses

This project is a Smart Search Tool designed to help users find the most relevant free courses available on the **Analytics Vidhya** platform. The tool supports both keyword-based and natural language queries, offering an intuitive interface and efficient search capabilities.

## Features

- **Semantic Search:** Uses state-of-the-art embeddings to provide accurate and relevant search results.
- **Natural Language Queries:** Allows users to search using conversational language.
- **Interactive UI:** Built with Streamlit for easy user interaction.
- **Fast and Efficient:** Powered by FAISS for high-speed similarity searches.
- **Scraped Data:** Includes course titles, descriptions, and URLs.

## Tech Stack

- **Programming Language:** Python
- **Libraries:**
  - `LangChain`
  - `sentence-transformers`
  - `FAISS`
  - `Streamlit`
  - `BeautifulSoup` and `requests` (for web scraping)

## Installation and Setup

### Prerequisites

- Python 3.7 or later
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Vickym78/Search-tool.git
   cd Search-tool
    pip install -r requirements.txt
   streamlit run app.py
       ```
