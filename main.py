import os
import uuid
import shutil
import logging
import boto3
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from RAG import (
    load_resume,
    split_text,
    load_and_embed_chunks,
    load_index,
    initialize_interview,
    chat_with_interviewer,
)
from interview_manager import InterviewManager
from report import evaluate_interview_transcript

# Load env variables
load_dotenv()
app = FastAPI()

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="eu-central-1",
)

bucket_name = os.getenv("S3_BUCKET_NAME")

API_KEY = os.getenv("API_KEY")

# In-memory storage for demo
USER_CREDITS = {}  # user_id -> credits
USER_VECTORSTORE = {}  # user_id -> vectorstore
USER_HISTORY = {}
USER_INTERVIEW_MANAGER = {}

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency: validate user and initialize credits from DB header (only first time)
def check_user(
    api_key: str = Header(..., alias="api-key"),
    user_id: str = Header(..., alias="user-id"),
    user_credits: int = Header(None, alias="user-credits"),  # DB-provided value
):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Initialize credits if new user
    if user_id not in USER_CREDITS:
        if user_credits is None:
            raise HTTPException(
                status_code=400, detail="Missing user credits for new user."
            )
        USER_CREDITS[user_id] = user_credits
        logger.info(f"Initialized credits for {user_id}: {user_credits}")

    return user_id


# Helper to normalize strings for S3 paths
def normalize_string(s):
    return s.strip().lower().replace(" ", "_").replace("/", "_")


# Helper to consume credits safely
def consume_credit(user_id, amount=1):
    if USER_CREDITS[user_id] < amount:
        raise HTTPException(status_code=402, detail="User Credits Exhausted")
    USER_CREDITS[user_id] -= amount
    return USER_CREDITS[user_id]


# Models
class JobContext(BaseModel):
    role_title: str
    company: str
    role_description: str
    required_skills: str
    years_of_experience: int
    interview_timing: int


class UserReply(BaseModel):
    user_input: str


# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Interview RAG API!"}


@app.post("/uploadResume")
async def upload_resume(
    resume: UploadFile = File(...), user_id: str = Depends(check_user)
):
    temp_dir = "temp_resumes"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    temp_filepath = os.path.join(temp_dir, temp_filename)

    with open(temp_filepath, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)

    documents = load_resume(temp_filepath)
    os.remove(temp_filepath)

    if not documents:
        return {
            "message": "No documents found in resume.",
            "remaining_credits": USER_CREDITS[user_id],
        }

    chunks = split_text(documents)
    vectorstore = load_and_embed_chunks(chunks)
    USER_VECTORSTORE[user_id] = vectorstore

    return {
        "message": "Resume uploaded and processed.",
        "remaining_credits": USER_CREDITS[user_id],
    }


@app.post("/startInterview")
async def start_interview_api(job: JobContext, user_id: str = Depends(check_user)):
    # Check credits before LLM call
    if USER_CREDITS[user_id] <= 0:
        raise HTTPException(status_code=402, detail="User Credits Exhausted")

    # Load user-specific vectorstore
    vectorstore = USER_VECTORSTORE.get(user_id)
    if vectorstore is None:
        vectorstore = load_index()
        USER_VECTORSTORE[user_id] = vectorstore

    # Try LLM call and rollback credits on failure
    before = USER_CREDITS[user_id]
    try:
        history = []
        question, history = initialize_interview(
            vectorstore, job.dict(), history=history
        )
        USER_HISTORY[user_id] = history
        USER_INTERVIEW_MANAGER[user_id] = InterviewManager(job.interview_timing)
    except Exception as e:
        USER_CREDITS[user_id] = before
        logger.exception(f"Error starting interview for user {user_id}")
        raise HTTPException(status_code=502, detail="Interview backend error.") from e

    # Deduct credit only after success
    remaining = consume_credit(user_id, 1)
    logger.info(f"User {user_id} started interview. Remaining credits: {remaining}")

    return {"interviewer": question, "remaining_credits": remaining}


@app.post("/chat")
async def chat_with_bot(
    reply: UserReply, job: JobContext, user_id: str = Depends(check_user)
):
    vectorstore = USER_VECTORSTORE.get(user_id)
    if vectorstore is None:
        vectorstore = load_index()
        USER_VECTORSTORE[user_id] = vectorstore

    history = USER_HISTORY.get(user_id, [])

    interview_manager = USER_INTERVIEW_MANAGER.get(user_id)
    current_phase = "resume"
    if interview_manager:
        current_phase = interview_manager.get_current_phase()

    if interview_manager.is_interview_over():
        response, history = chat_with_interviewer(
            reply.user_input,
            job.dict(),
            vectorstore,
            history,
            userid=user_id,
            phase= "closed",
        )
        USER_HISTORY[user_id] = history
        return {
            "interviewer": "I think we've covered everything. Thank you for your time! Have a great day!",
            "phase": "closed",
            "remaining_credits": USER_CREDITS[user_id],
        }

    response, history = chat_with_interviewer(
        reply.user_input,
        job.dict(),
        vectorstore,
        history,
        userid=user_id,
        phase=current_phase,
    )

    USER_HISTORY[user_id] = history
    return {
        "interviewer": response,
        "phase": current_phase,
        "remaining_credits": USER_CREDITS[user_id],
    }


@app.post("/evaluateInterview")
async def evaluate_interview(
    job: JobContext, 
    user_id: str = Depends(check_user),
    interview_date: Optional[str] = Header(None, alias="interview-date"), 
):
    company = normalize_string(job.company)
    role = normalize_string(job.role_title)

    prefix = f"interviews/{user_id}/{company}/{role}/"

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response or not response["Contents"]:
        return {"error": "No interview data found for this job context."}

    # If date provided, filter filenames
    files = [obj["Key"] for obj in response["Contents"]]
    if interview_date:
        files = [f for f in files if f.startswith(f"{prefix}{interview_date}")]
        if not files:
            return {"error": f"No file found for date {interview_date}"}
        selected_file = sorted(files, reverse=True)[0]  # pick latest on that date
    else:
        # fallback to latest
        selected_file = sorted(
            response["Contents"], key=lambda x: x["LastModified"], reverse=True
        )[0]["Key"]

    obj = s3.get_object(Bucket=bucket_name, Key=selected_file)
    interview_data = json.loads(obj["Body"].read().decode("utf-8"))

    report = evaluate_interview_transcript(interview_data)

    return {
        "message": "Interview data retrieved successfully",
        "interview_metadata": {
            "user_id": user_id,
            "job_role": job.role_title,
            "company": job.company,
            "interview_time": interview_data.get("timestamp", "unknown"),
            "file": selected_file,
        },
        "report": report,
    }

