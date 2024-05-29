from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

# Define the directory to save PDF files
PDF_DIR = "uploaded_pdfs"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


async def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_question", methods=["POST"])
def process_question():
    if request.method == 'POST':
        data = request.json
        user_question = data.get('question')

        if user_question:
            response = asyncio.run(user_input(user_question))
            return jsonify({"response": response})
        else:
            return jsonify({"error": "No question provided"}), 400
    else:
        return jsonify({"error": "Only POST requests are supported"}), 405


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "No PDF file selected"}), 400

    pdf_path = os.path.join(PDF_DIR, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Process the uploaded PDF
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_chunks = get_text_chunks(text)
    asyncio.run(get_vector_store(text_chunks))

    return jsonify({"message": "PDF uploaded and processed successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True)
