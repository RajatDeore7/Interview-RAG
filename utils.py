import boto3
import json
import os
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
    s3_key = f"interviews/{user_id}.json"

    try:
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        existing_data = json.loads(response["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        existing_data = []

    interview_data["timestamp"] = datetime.utcnow().isoformat()
    existing_data.append(interview_data)

    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=json.dumps(serialize_chat_history(existing_data), indent=2),
        ContentType="application/json",
    )
