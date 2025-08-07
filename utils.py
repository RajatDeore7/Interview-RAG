import boto3
import json
import os
import uuid
from datetime import datetime
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="eu-central-1",
)

bucket_name = os.getenv("S3_BUCKET_NAME")


def normalize_string(s):
    return s.strip().lower().replace(" ", "_").replace("/", "_")


def serialize_chat_history(data):
    serialized = []
    for entry in data:
        if isinstance(entry, AIMessage):
            serialized.append({"type": "ai", "text": entry.content})
        elif isinstance(entry, HumanMessage):
            serialized.append({"type": "human", "text": entry.content})
        else:
            serialized.append({"type": "unknown", "text": str(entry)})
    return serialized


def store_interview_to_s3(user_id, interview_data):
    # Serialize conversation history
    if "conversation_history" in interview_data:
        interview_data["conversation_history"] = serialize_chat_history(
            interview_data["conversation_history"]
        )

    # Add timestamp
    timestamp = datetime.utcnow().isoformat()
    interview_data["timestamp"] = timestamp

    # Extract job info
    job_context = interview_data.get("job_context", {})
    company = normalize_string(job_context.get("company", "unknown_company"))
    role = normalize_string(job_context.get("role_title", "unknown_role"))

    # Optional: Add UUID or run id for duplicates
    run_id = str(uuid.uuid4())[:8]

    # Filename & path
    filename = f"{timestamp.replace(':', '-')}_{run_id}.json"
    s3_key = f"interviews/{user_id}/{company}/{role}/{filename}"

    # Upload
    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=json.dumps(interview_data, indent=2),
        ContentType="application/json",
    )
    print(f"✅ Saved interview to S3: s3://{bucket_name}/{s3_key}")
