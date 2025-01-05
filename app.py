import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Cache embeddings and vector store for performance
@st.cache_resource
def create_vector_store(courses_data):
    """
    Create a FAISS vector index using LangChain.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    docs = [Document(page_content=course['description'], metadata=course) for course in courses_data]
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def fetch_courses():
    """
    Scrape free courses from the Analytics Vidhya platform.
    """
    BASE_URL = "https://courses.analyticsvidhya.com/collections/courses"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(BASE_URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    courses = []
    course_blocks = soup.find_all("div", class_="course-block")
    for block in course_blocks:
        try:
            title = block.find("h4", class_="course-title").text.strip()
            description = block.find("p", class_="course-description").text.strip()
            link = block.find("a", href=True)["href"]
            url = f"https://courses.analyticsvidhya.com{link}"
            courses.append({"title": title, "description": description, "url": url})
        except Exception as e:
            st.error(f"Error parsing course: {e}")
    return courses

@st.cache_data
def get_course_data():
    """
    Retrieve and cache course data.
    """
    return fetch_courses()

# Fetch course data and create vector store
courses_data = get_course_data()
vector_store = create_vector_store(courses_data)

def search_courses(query, k=5):
    """
    Search for relevant courses using LangChain and FAISS.
    """
    results = vector_store.similarity_search(query, k=k)
    return results

# Streamlit UI
st.title("Smart Course Search")
st.write("Search for free courses on Analytics Vidhya!")

query = st.text_input("Enter your search query", "")

if query:
    st.write("### Search Results")
    results = search_courses(query)
    if results:
        for idx, result in enumerate(results, start=1):
            metadata = result.metadata
            st.write(f"**{idx}.** {metadata['title']}\n{metadata['description']}\n[Learn More]({metadata['url']})")
    else:
        st.write("No results found!")

if st.checkbox("Show all courses"):
    st.write("### Available Courses")
    for course in courses_data:
        st.write(f"**{course['title']}**\n{course['description']}\n[Learn More]({course['url']})")
