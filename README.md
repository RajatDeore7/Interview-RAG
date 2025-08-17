# 🧠 Resume Interview RAG API

An AI-powered interview simulation app using FastAPI, LangChain, FAISS, HuggingFace, and Gemini (Google Generative AI).

---

## 🚀 Features

- Upload and parse a PDF resume
- Generate embeddings using HuggingFace + FAISS
- Simulate contextual job interviews using Gemini 2.5 Pro
- Credit-limited API access via API keys
- FastAPI backend ready for deployment

---

## 📁 Project Structure

```text
RAG/
├── 📁 .git/ 🚫 (auto-hidden)
├── 📁 __pycache__/ 🚫 (auto-hidden)
├── 📁 faiss_index/ 🚫 (auto-hidden)
├── 📁 temp_resumes/ 🚫 (auto-hidden)
├── 📁 venv/ 🚫 (auto-hidden)
├── 📄 .DS_Store 🚫 (auto-hidden)
├── 🔒 .env 🚫 (auto-hidden)
├── 🚫 .gitignore
├── 🐍 RAG.py
├── 📖 README.md
├── 🐍 interview_manager.py
├── 🐍 main.py
├── 🐍 report.py
├── 📄 requirements.txt
├── 🐚 start.sh
└── 🐍 utils.py
```

---

## 🛠 Installation

```bash
git clone https://github.com/RajatDeore7/resume-rag-api.git
cd resume-rag-api

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
🔐 Environment Variables (.env)
Create a .env file in the root with the following:

API_KEY=your_test_key
GEMINI_KEY=your_google_gemini_api_key
▶️ Running Locally

bash start.sh
Open your browser: http://localhost:8000/docs to access the Swagger UI.

🚀 Deploying to Railway
Push this code to a GitHub repo

Go to https://railway.app

Click New Project > Deploy from GitHub

Add environment variables (API_KEY, GEMINI_KEY) in Settings > Variables

Done! 🎉

📬 API Endpoints
POST /uploadResume - Upload a resume PDF

POST /startInterview - Start the interview session

POST /chat - Send a response and get next question

✅ To Do
Add user auth system (optional)

Add frontend (React or Next.js)

Dockerize the app

🧠 Built With
FastAPI

LangChain

Google Gemini

FAISS

Railway

📄 License
MIT License

Let me know if you'd like a `Docker

```
