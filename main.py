from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any
import os
import requests
from pathlib import Path
import json
import re
import traceback


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 作业演示先用 * 最省事
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== Paths ======
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# Serve static files (css/js/images) under /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

Mode = Literal["chat", "rewrite_shorter", "rewrite_politer", "rewrite_confident"]

class ChatRequest(BaseModel):
    message: str
    tone: Optional[str] = "Calm"
    scenario: Optional[str] = "general"
    mode: Optional[Mode] = "chat"

@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing frontend/index.html at: {index_path}")
    return FileResponse(str(index_path))


# ========== Prompt pieces ==========
def build_system_instruction(tone: str, scenario: str) -> str:
    return f"""
You are Echo — a conversation rehearsal assistant for people with social anxiety.

Core rules:
- Be kind, calm, and practical. No judgement, no diagnosis, no lecturing.
- Do NOT ask the user any questions.
- Do NOT guide the conversation (the UI already provides intent).
- Always provide ready-to-use sentences the user can copy.
- Keep it SHORT and concrete. Prefer 1 sentence for the main reply.
- Use simple everyday English. Avoid long paragraphs.
- Do NOT mention you are an AI or mention "prompt" or "policy".
- Avoid generic coaching phrases like “What would you like to…” / “What kind of…”

Context:
- Tone = {tone}
- Scenario = {scenario}

Output requirement (MUST follow):
Return ONLY valid JSON. No markdown. No extra text.

JSON schemas:
1) mode=chat
{{
  "reply": "string",
  "options": ["string", "string", "string"]
}}

2) rewrite modes
{{
  "rewrite": "string"
}}

Extra requirements for mode=chat:
- "reply" = best default phrasing (1 sentence).
- "options" = 3 different alternatives (each 1 sentence), varying slightly in formality/softness.
""".strip()


def build_user_content(message: str, mode: str) -> str:
    if mode == "chat":
        task = "Provide Echo's best reply and 3 alternative phrasings."
    elif mode == "rewrite_shorter":
        task = "Rewrite the user's message to be shorter without changing meaning."
    elif mode == "rewrite_politer":
        task = "Rewrite the user's message to be more polite (not overly apologetic)."
    elif mode == "rewrite_confident":
        task = "Rewrite the user's message to sound more confident and clear."
    else:
        task = "Provide Echo's best reply and 3 alternative phrasings."

    return f"""{task}

User message:
{message}
""".strip()


# ========== Gemini caller ==========
def call_gemini(system_text: str, user_text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY environment variable.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]}
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 420
        }
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not reach Gemini: {repr(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Gemini HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()

    # Merge all parts
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise ValueError("Empty Gemini response")
        return text
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected Gemini response structure: {data}")


def extract_json(text: str) -> Dict[str, Any]:
    """
    If model adds extra text, try to extract the first JSON object.
    """
    text = text.strip()

    # Direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object in the string
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise HTTPException(status_code=500, detail=f"Model did not return JSON. Raw: {text[:500]}")
    try:
        return json.loads(m.group(0))
    except Exception:
        raise HTTPException(status_code=500, detail=f"Bad JSON from model. Raw: {text[:500]}")


@app.post("/chat")
def chat(req: ChatRequest):
    tone = req.tone or "Calm"
    scenario = req.scenario or "general"
    mode = req.mode or "chat"

    system_text = build_system_instruction(tone, scenario)
    user_text = build_user_content(req.message, mode)

    raw = call_gemini(system_text, user_text)
    obj = extract_json(raw)

    # Normalize response for frontend
    if mode == "chat":
        reply = (obj.get("reply") or "").strip()
        options = obj.get("options") or []

        # Ensure options length = 3
        options = [str(x).strip() for x in options][:3]
        while len(options) < 3:
            options.append("")

        if not reply:
            raise HTTPException(status_code=500, detail=f"Empty reply from model. Raw: {raw[:500]}")
        return {"reply": reply, "options": options}

    # rewrite modes
    rewrite = (obj.get("rewrite") or "").strip()
    if not rewrite:
        raise HTTPException(status_code=500, detail=f"Empty rewrite from model. Raw: {raw[:500]}")
    return {"reply": rewrite}

