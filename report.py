import json
import re
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage


def clean_content(raw: str) -> str:
    # Extract content='...' or "..." using regex
    match = re.search(r"content=['\"](.*?)['\"]\)?$", raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def evaluate_interview_transcript(interview_data: dict):
    print("Received interview_data keys:", interview_data.keys())

    job_context = interview_data["job_context"]
    history = interview_data["conversation_history"]

    role_title = job_context.get("role_title")
    company = job_context.get("company")
    required_skills = job_context.get("required_skills")
    role_description = job_context.get("role_description")
    years_of_experience = job_context.get("years_of_experience")

    evaluation_results = []

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key="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",
        temperature=0.3,
        max_tokens=1000,
    )

    current_question = None

    for item in history:
        text = item["text"]

        try:
            parsed = eval(text)
        except Exception:
            continue

        if parsed.get("role") == "assistant":
            current_question = clean_content(str(parsed.get("content", "")))

        elif parsed.get("role") == "user" and current_question:
            user_answer = clean_content(str(parsed.get("content", "")))

            system_msg = AIMessage(
                content=(
                    f"You are an expert technical interviewer evaluating a candidate's response for the role of {role_title} at {company}.\n"
                    f"Required Skills: {required_skills}\n"
                    f"Role Description: {role_description}\n"
                    f"Years of Experience Required: {years_of_experience}\n\n"
                    "Evaluate the following Q&A:\n"
                    "- Check if the answer is relevant and complete\n"
                    "- Identify which required skills are demonstrated\n"
                    "- List missing skills/concepts\n"
                    "- Score the answer from 1 to 10 with justification\n\n"
                    "Return a JSON object with the following fields:\n"
                    "`relevant`, `skills_covered`, `missing_elements`, `score`, `feedback`"
                    "Add final verdict as 'pass' or 'fail' based on the score.\n\n at the end of the JSON, include a 'raw_output' field with the original answer.\n\n"
                )
            )

            user_msg = HumanMessage(
                content=f"Question: {current_question}\nCandidate Answer: {user_answer}"
            )

            try:
                response = llm.invoke([system_msg, user_msg])
                parsed_eval = response.content
            except Exception:
                parsed_eval = {
                    "error": "Failed to parse or incomplete JSON",
                    "raw_output": response.content,
                }

            evaluation_results.append(
                {
                    "question": current_question,
                    "answer": user_answer,
                    "evaluation": parsed_eval,
                }
            )

            current_question = None

    return evaluation_results
