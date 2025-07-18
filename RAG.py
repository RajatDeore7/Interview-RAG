import os
import tempfile
import shutil
import boto3 as b3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_ollama import OllamaLLM

# Globals for simple session
message_history = []
vectorstore = None
load_dotenv()


# Load resume PDF
def load_resume(file_path):
    if file_path:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    print("No file path provided.")
    return []


# Split text into chunks
def split_text(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter.split_documents(docs)


# Embed chunks and save to FAISS
def load_and_embed_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore.save_local(temp_dir)

        # Upload all files in temp_dir to S3
        s3 = b3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name="eu-central-1",
        )
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set.")

        for root, _, files in os.walk(temp_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                s3_key = os.path.relpath(file_path, temp_dir)
                s3.upload_file(file_path, bucket_name, f"faiss_index/{s3_key}")

    return vectorstore


# Load existing FAISS index
def load_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    # Load FAISS index from S3 directly into a temp directory
    s3 = b3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name="eu-central-1",
    )

    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    with tempfile.TemporaryDirectory() as temp_dir:
        faiss_file = os.path.join(temp_dir, "index.faiss")
        pkl_file = os.path.join(temp_dir, "index.pkl")

        s3.download_file(bucket_name, "faiss_index/index.faiss", faiss_file)
        s3.download_file(bucket_name, "faiss_index/index.pkl", pkl_file)

        # Load the FAISS index directly from temp directory
        vectorstore = FAISS.load_local(
            folder_path=temp_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore


# Similarity search from vectorstore
def get_resume_context(vectorstore, query="candidate introduction", k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


# Start the interview
def initialize_interview(vstore, job_context: dict):
    global message_history, vectorstore
    vectorstore = vstore
    message_history = []

    resume_context = get_resume_context(vectorstore, "candidate introduction")

    # Create job context string
    job_context_str = (
        f"Company: {job_context.get('company')}\n"
        f"Role Title: {job_context.get('role_title')}\n"
        f"Required Skills: {job_context.get('required_skills')}\n"
        f"Role Description: {job_context.get('role_description')}\n"
        f"Years of Experience: {job_context.get('years_of_experience')}"
    )

    # System message includes both contexts
    system_prompt = (
        "You are an experienced interviewer conducting a job interview.\n"
        "Follow these rules:\n"
        "1. Ask only ONE question at a time.\n"
        "2. Keep the question short and clear.\n"
        "3. Start the interview by greeting the candidate and asking them to introduce themselves.\n"
        "4. After introduction, ask role-related questions based on:\n"
        "- Candidate's resume\n"
        "- Job description\n"
        "- Years of experience and skills.\n"
        "5. Do NOT make assumptions about the candidate's experience before they introduce themselves.\n"
        "6. Do NOT include long context in the question. Keep it conversational.\n"
        f"\nJob Information:\n{job_context_str}"
    )

    user_prompt = (
        f"This is the candidate's resume for reference (do NOT summarize it in your first question):\n{resume_context}\n\n"
        "Now, start the interview with a friendly greeting and ask them to introduce themselves."
    )

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # llm = OllamaLLM(model="llama3")
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_KEY"), temperature=0.7
    # )
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_KEY"),
        temperature=0.7,
        max_tokens=1000,
    )
    ai_reply = llm.invoke(full_prompt)

    message_history.append({"role": "system", "content": system_prompt})
    message_history.append({"role": "user", "content": user_prompt})
    message_history.append({"role": "assistant", "content": ai_reply})

    return ai_reply


# Chat with the interviewer
def chat_with_interviewer(user_input, job_context):
    global message_history, vectorstore

    resume_reference = get_resume_context(vectorstore, user_input, k=3)

    # Build job context string
    job_context_str = (
        f"Company: {job_context.get('company')}\n"
        f"Role Title: {job_context.get('role_title')}\n"
        f"Required Skills: {job_context.get('required_skills')}\n"
        f"Role Description: {job_context.get('role_description')}\n"
        f"Years of Experience: {job_context.get('years_of_experience')}"
    )

    # Append user message
    message_history.append({"role": "user", "content": user_input})

    # Prepare full conversation context
    conversation = ""
    for msg in message_history:
        conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Compose full prompt
    full_prompt = (
        f"{conversation}\n\n"
        f"(Additional Context)\nJob Info:\n{job_context_str}\n"
        f"Relevant Resume Info:\n{resume_reference}\n"
        "Ask the next short, relevant question based on this conversation. Do not repeat previous questions."
    )

    # llm = OllamaLLM(model="llama3")
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_KEY"), temperature=0.7
    # )
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_KEY"),
        temperature=0.7,
        max_tokens=1000,
    )
    ai_reply = llm.invoke(full_prompt)

    # Save assistant reply
    message_history.append({"role": "assistant", "content": ai_reply})

    return ai_reply
