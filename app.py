import requests
from bs4 import BeautifulSoup

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
    course_blocks = soup.find_all("div", class_="course-block")  # Adjust based on site structure
    for block in course_blocks:
        try:
            title = block.find("h4", class_="course-title").text.strip()
            description = block.find("p", class_="course-description").text.strip()
            link = block.find("a", href=True)["href"]
            url = f"https://courses.analyticsvidhya.com{link}"
            courses.append({"title": title, "description": description, "url": url})
        except Exception as e:
            print(f"Error parsing course: {e}")
    return courses

# Fetch the course data
courses_data = fetch_courses()
print(f"Fetched {len(courses_data)} courses.")
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Generate embeddings using OpenAI or another model
embeddings = OpenAIEmbeddings()  # Requires OpenAI API Key in the environment

# Create vector database with FAISS
descriptions = [course['description'] for course in courses_data]
vector_store = FAISS.from_texts(descriptions, embeddings)

# Save the vector database locally
vector_store.save_local("course_embeddings")
def search_courses(query, k=5):
    # Load the saved vector store
    vector_store = FAISS.load_local("course_embeddings", embeddings)
    results = vector_store.similarity_search(query, k=k)
    return results
import gradio as gr

# Define the Gradio function for search
def query_interface(input_text):
    results = search_courses(input_text)
    output = "\n\n".join([f"{i+1}. {result['text']}" for i, result in enumerate(results)])
    return output

# Create a Gradio app
interface = gr.Interface(
    fn=query_interface,
    inputs=gr.Textbox(label="Search for Free Courses"),
    outputs=gr.Textbox(label="Search Results"),
    title="Smart Course Search",
    description="Search for the most relevant free courses on Analytics Vidhya."
)
