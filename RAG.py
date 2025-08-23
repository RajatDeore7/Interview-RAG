import os
import tempfile
import boto3 as b3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from datetime import datetime
from utils import store_interview_to_s3

# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_ollama import OllamaLLM

# Globals for simple session
vectorstore = None
load_dotenv()

total_input_tokens = 0
total_output_tokens = 0

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
def initialize_interview(vstore, job_context: dict, history=None):

    if history is None:
        history = []

    global vectorstore
    vectorstore = vstore

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
        api_key="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",
        temperature=0.7,
        max_tokens=1000,
    )
    ai_reply = llm.invoke(full_prompt)

    history.append({"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": ai_reply})

    return ai_reply, history


# Chat with the interviewer
def chat_with_interviewer(
    user_input, job_context, vstore, history, userid, phase="resume"
):
    global vectorstore
    vectorstore = vstore
    global total_input_tokens, total_output_tokens

    resume_reference = get_resume_context(vectorstore, user_input, k=3)

    # Build job context string
    job_context_str = (
        f"Company: {job_context.get('company')}\n"
        f"Role Title: {job_context.get('role_title')}\n"
        f"Required Skills: {job_context.get('required_skills')}\n"
        f"Role Description: {job_context.get('role_description')}\n"
        f"Years of Experience: {job_context.get('years_of_experience')}"
    )

    # Phase-specific instruction
    if phase == "wrapup":
        phase_instruction = """
        You are in the WRAP-UP phase.  
        Your goal: close the interview politely and professionally.

        Rules:
        1. Thank the candidate for their time and participation.  
        2. Offer them the chance to ask questions about the company or role.  
        3. Provide a warm closing statement, leaving a positive impression.  
        4. Do not ask new technical, behavioral, or resume-based questions.  
        5. Keep the tone friendly and conversational.
        """
    elif phase == "behavioral":
        phase_instruction = """
        You are in the BEHAVIORAL phase.  
        Your goal: understand the candidate’s values, personality, and soft skills.

        Rules:
        1. Ask about teamwork, leadership, communication, handling conflict, and adaptability.  
        2. Use the STAR method (Situation, Task, Action, Result) as a guide—encourage detailed answers.  
        3. Avoid technical or resume-only questions in this phase.  
        4. If an answer is too short, ask for specific examples from work, school, or projects.
        """
    elif phase == "role based technical":
        phase_instruction = """
        You are in the ROLE-BASED TECHNICAL phase.  
        Your goal: assess the candidate’s technical depth.

        Rules:
        1. Ask job-specific technical questions based on the provided role, required skills, and experience level.  
        2. Use real-world, scenario-based questions whenever possible.  
        3. Probe for problem-solving process, reasoning, and trade-offs—not just definitions.  
        4. If the candidate’s answer is vague, follow up with a request for concrete examples or code-level explanation.  
        5. Avoid behavioral or resume-only questions in this phase.
        6. Ask one or two DSA and OOPs related question based on the required skills.
        """
    else:
        phase_instruction = """
        You are in the RESUME-BASED phase.  
        Your goal: explore the candidate’s background in detail.

        Rules:
        1. Focus only on topics mentioned in the resume (education, projects, internships, certifications, achievements).  
        2. Ask open-ended questions that require explanations, examples, or stories—not just yes/no answers.  
        3. Avoid asking technical coding or behavioral questions in this phase.  
        4. Show curiosity and encourage elaboration (e.g., “Tell me more about…”).  
        5. Use resume context to create follow-up questions that dig deeper into specific experiences.
        """

    # Append user message
    history = list(history)  # shallow copy
    history.append({"role": "user", "content": user_input})

    # Prepare full conversation context
    conversation = ""
    for msg in history:
        conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Compose full prompt
    full_prompt = f"""
        You are a professional interviewer for the role described below.

        Job Info:
        {job_context_str}

        Relevant Resume Info:
        {resume_reference}

        Interview Phase Instructions:
        {phase_instruction}

        Conversation so far:
        {conversation}

        Your task:
        1. If the candidate wants to end the interview (mentions "stop", "end", or "quit"), respond politely and conclude.
        2. Otherwise, ask the next relevant question.
        3. If the candidate's last answer was short, vague, or only yes/no, follow up with a deeper question, 
           asking for examples, projects, or real-world applications.
        4. Avoid repeating previous questions.
        5. Keep questions concise and clear.
        6. Always ask only ONE question at a time.
        7. If the candidate's last answer was good, acknowledge it and ask a follow-up question.
        8. If the candidate's last answer was not good, ask a more specific question to clarify.
        9. If the candidate ask to repeat the question, so repeat the last question.
    """

    # llm = OllamaLLM(model="llama3")
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_KEY"), temperature=0.7
    # )
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key="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",
        temperature=0.7,
        max_tokens=1000,
    )
    ai_reply = llm.invoke(full_prompt)
    # print(f"AI Reply: {ai_reply}")
    
    #get total tokens used
    if hasattr(ai_reply, "usage_metadata") and ai_reply.usage_metadata:
        input_tokens = ai_reply.usage_metadata.get("input_tokens", 0)
        output_tokens = ai_reply.usage_metadata.get("output_tokens", 0)
        total_tokens = ai_reply.usage_metadata.get("total_tokens", 0)
    else:
        token_info = ai_reply.response_metadata.get("token_usage", {})
        input_tokens = token_info.get("prompt_tokens", 0)
        output_tokens = token_info.get("completion_tokens", 0)
        total_tokens = token_info.get("total_tokens", 0)
        
    #Count total tokens used
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    # Save assistant reply
    history.append({"role": "assistant", "content": ai_reply})

    # Store interview data to S3 after phase wrap-up
    if phase == "closed":
        interview_data = {
            "job_context": job_context,
            "conversation_history": history,
            "timestamp": datetime.utcnow().isoformat(),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        }
        print(f"Storing interview data for user {userid} to S3...")
        store_interview_to_s3(userid, interview_data)

    return ai_reply, history
