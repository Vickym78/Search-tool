import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os


os.environ["OPENAI_API_KEY"] = "api-key"

# Function to fetch courses
def fetch_courses():
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


@st.cache
def get_course_data():
    return fetch_courses()

courses_data = get_course_data()

# Generate embeddings using OpenAI
@st.cache(allow_output_mutation=True)
def create_vector_store(courses_data):
    embeddings = OpenAIEmbeddings()
    descriptions = [course['description'] for course in courses_data]
    vector_store = FAISS.from_texts(descriptions, embeddings)
    vector_store.save_local("course_embeddings")
    return vector_store

vector_store = create_vector_store(courses_data)

# Function to search courses
def search_courses(query, k=5):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("course_embeddings", embeddings)
    results = vector_store.similarity_search(query, k=k)
    return results

# Streamlit App
st.title("Smart Course Search")
st.write("Search for free courses on Analytics Vidhya!")

# Input from user
query = st.text_input("Enter your search query", "")

if query:
    st.write("### Search Results")
    results = search_courses(query)
    if results:
        for idx, result in enumerate(results, start=1):
            st.write(f"**{idx}.** {result['text']}")
    else:
        st.write("No results found!")

# Optionally, display the fetched course data
if st.checkbox("Show all courses"):
    st.write("### Available Courses")
    for course in courses_data:
        st.write(f"**{course['title']}**\n{course['description']}\n[Learn More]({course['url']})")
