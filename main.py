import os
import uuid
import shutil
import traceback
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from pydantic import BaseModel
from RAG import (
    load_resume,
    split_text,
    load_and_embed_chunks,
    load_index,
    initialize_interview,
    chat_with_interviewer,
)
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_Credits = {}

vectorstore = None

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_api_key(
    api_key: str = Header(..., alias="api-key"),
    user_credits: int = Header(..., alias="user-credits"),
):
    # Validate the API Key with api_key
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # If this is the first time we've seen the key, initialize its credits
    if api_key not in API_KEY_Credits:
        API_KEY_Credits[api_key] = user_credits
        print(f"Initialized credits for {api_key}: {user_credits}")

    # Get current credits
    credits = API_KEY_Credits.get(api_key, 0)

    # If exhausted, reject the request
    if credits <= 0:
        raise HTTPException(status_code=401, detail="User Credits Exhausted")

    return api_key


# test response
@app.get("/")
async def root():
    """
    Test endpoint to check if the server is running.
    """
    return {"message": "Welcome to the Interview RAG API!"}


@app.post("/uploadResume")
async def upload_resume(resume: UploadFile = File(...)):
    """
    Upload and process the resume PDF file from user input.
    """
    global vectorstore

    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    temp_dir = "temp_resumes"
    os.makedirs(temp_dir, exist_ok=True)  # <== Ensure the directory exists
    temp_filepath = os.path.join(temp_dir, temp_filename)

    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    # Process resume
    documents = load_resume(temp_filepath)
    if not documents:
        return {"message": "No documents found in resume."}

    chunks = split_text(documents)
    vectorstore = load_and_embed_chunks(chunks)

    os.remove(temp_filepath)  # Optional cleanup

    return {"message": "Resume uploaded and processed."}


class JobContext(BaseModel):
    role_title: str
    company: str
    role_description: str
    required_skills: str
    years_of_experience: int


@app.post("/startInterview")
async def start_interview_api(job: JobContext, api_key: str = Depends(check_api_key)):
    """
    Initialize chat and return first interview question.
    """
    API_KEY_Credits[api_key] -= 1
    print("Remaining Credits:", API_KEY_Credits[api_key])

    global vectorstore

    if vectorstore is None:
        vectorstore = load_index()

    question = initialize_interview(vectorstore, job.dict())
    return {"interviewer": question, "remaining_credits": API_KEY_Credits[api_key]}


class UserReply(BaseModel):
    user_input: str


@app.post("/chat")
async def chat_with_bot(reply: UserReply, job: JobContext):
    """
    Process user answer and return next AI question.
    """
    response = chat_with_interviewer(reply.user_input, job.dict())
    return {"interviewer": response}
