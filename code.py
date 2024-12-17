import os
import fitz  # PyMuPDF for PDF reading
import faiss
import requests
import threading
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ------------------ Step 0: Environment Setup ------------------
# Securely fetch API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Error: OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------ Step 1: Download PDFs from URLs ------------------
def download_pdf_from_url(url, save_path):
    """Download a PDF file from a URL and save it locally with error handling."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded PDF: {url}")
        else:
            print(f"Failed to download PDF (Status: {response.status_code}): {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def download_pdfs_multithreaded(pdf_links):
    """Download multiple PDFs using multithreading."""
    threads = []
    for idx, link in enumerate(pdf_links):
        save_path = f"pdf_{idx + 1}.pdf"
        thread = threading.Thread(target=download_pdf_from_url, args=(link, save_path))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    return [f"pdf_{i + 1}.pdf" for i in range(len(pdf_links))]

# ------------------ Step 2: PDF Text Extraction ------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        pdf = fitz.open(pdf_path)
        for page in pdf:
            text += page.get_text()
        pdf.close()
        print(f"Extracted text from: {pdf_path}")
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

# ------------------ Step 3: Text Chunking ------------------
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split the text into chunks with overlap for better embedding performance."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# ------------------ Step 4: Embedding and FAISS Vector Store ------------------
def create_vector_store(chunks):
    """Embed text chunks and store them in a FAISS vector database."""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        print("Successfully created FAISS vector store.")
        return vector_store
    except Exception as e:
        raise Exception(f"Embedding or FAISS vector creation error: {e}")

# ------------------ Step 5: Query Pipeline with Context Limit ------------------
def query_pipeline(user_query, vector_store, max_context_length=3000):
    """Handle user query: Retrieve relevant chunks and generate LLM response."""
    try:
        # Perform similarity search
        relevant_docs = vector_store.similarity_search(user_query, k=3)
        
        # Limit context length to avoid token overflow
        context = ""
        for doc in relevant_docs:
            if len(context) + len(doc.page_content) > max_context_length:
                break
            context += doc.page_content + "\n"
        
        # Generate response using GPT-4
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt_template = PromptTemplate(
            template="""
            Context:
            {context}

            Question:
            {question}

            Provide a clear and accurate response based on the context.
            """,
            input_variables=["context", "question"]
        )
        prompt = prompt_template.format(context=context, question=user_query)
        response = llm.predict(prompt)
        return response
    except Exception as e:
        return f"Query processing error: {e}"

# ------------------ Step 6: Main Pipeline ------------------
def main():
    # List of PDF links (replace with your own URLs)
    pdf_links = [
        "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf"
          # Add more links if needed
    ]

    # Step 1: Download PDFs
    print("Downloading PDFs...")
    local_pdf_paths = download_pdfs_multithreaded(pdf_links)

    # Step 2: Extract and combine text from PDFs
    print("Extracting text from PDFs...")
    combined_text = ""
    for pdf_path in local_pdf_paths:
        combined_text += extract_text_from_pdf(pdf_path) + "\n"

    # Step 3: Split text into chunks
    print("Splitting text into chunks...")
    text_chunks = chunk_text(combined_text)

    # Step 4: Create FAISS vector store
    print("Creating vector store...")
    vector_store = create_vector_store(text_chunks)

    # Step 5: User interaction loop
    print("\nYou can now ask questions about the PDF content. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        response = query_pipeline(query, vector_store)
        print("\nAnswer:")
        print(response)

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()
