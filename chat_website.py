import os
import requests
from bs4 import BeautifulSoup
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# Set the OpenAI API Key (ensure it's added as an environment variable for security)
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# --------- Step 1: Fetch and Parse Website Content ---------
def fetch_website_content(url):
    """Fetch HTML content from a URL and extract clean text."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract readable text: remove script, style, and other unnecessary tags
            for tag in soup(["script", "style", "meta", "noscript"]):
                tag.decompose()
            clean_text = soup.get_text(separator="\n", strip=True)
            print("Successfully fetched and cleaned website content.")
            return clean_text
        else:
            raise Exception(f"Failed to fetch website content: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
        return ""

# --------- Step 2: Chunking Text ---------
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks with overlap for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# --------- Step 3: Create Vector Store ---------
def create_vector_store(chunks):
    """Embed text chunks and store them in a FAISS vector database."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    print("Created FAISS vector store with embeddings.")
    return vector_store

# --------- Step 4: Query and Response Generation ---------
def query_pipeline(user_query, vector_store):
    """Handle user queries by retrieving relevant content and generating a response."""
    # Perform similarity search
    relevant_docs = vector_store.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Prompt template for GPT-4
    prompt_template = PromptTemplate(
        template="""
        Context:
        {context}
        
        Question:
        {question}
        
        Provide a clear, accurate, and fact-based response using the context provided.
        """,
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context, question=user_query)
    
    # Generate response using GPT-4
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    response = llm.predict(prompt)
    return response

# --------- Step 5: Main Pipeline ---------
def main():
    # Input website URL
    url = input("Enter the website URL to chat with: ")
    print("\nFetching and processing website content...")
    
    # Step 1: Fetch website content
    website_text = fetch_website_content(url)
    if not website_text:
        print("Failed to retrieve content. Exiting...")
        return

    # Step 2: Split text into chunks
    print("\nSplitting website content into chunks...")
    text_chunks = chunk_text(website_text)

    # Step 3: Create FAISS vector store
    print("\nCreating vector store...")
    vector_store = create_vector_store(text_chunks)

    # Step 4: Interactive query loop
    print("\nYou can now ask questions about the website content. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        response = query_pipeline(user_query, vector_store)
        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main()
