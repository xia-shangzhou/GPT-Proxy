# GPT Proxy — GitHub-ready Project

This repository contains a production-ready example of the GPT API proxy described earlier (FastAPI + httpx). It supports parameter mapping (`max_tokens` → `max_output_tokens` / `max_completion_tokens`), streaming passthrough, basic auth via `X-API-KEY`, Docker, docker-compose, tests, and CI-friendly structure.

---

## Repository structure

```
gpt-proxy/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  └─ utils.py
├─ tests/
│  └─ test_map.py
└─ .github/
   └─ workflows/
      └─ python-app.yml
```

---

## .gitignore

```
__pycache__/
*.pyc
.env
.env.*
*.sqlite3
.idea/
.vscode/
dist/
build/
.env.local
"""
```

---

## README.md

````md
# GPT Proxy (FastAPI)

A lightweight API proxy to convert legacy GPT-4o-style client requests to GPT-5 / Responses API format. Supports streaming passthrough, parameter mappings, basic authentication, Docker, and tests.

## Quick start

1. Copy `.env.example` to `.env` and fill values.

```bash
cp .env.example .env
# set OPENAI_API_KEY and PROXY_API_KEY
````

2. Run locally with docker-compose:

```bash
docker-compose up --build
```

3. Example request:

```bash
curl -X POST "http://localhost:8000/proxy" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <your-proxy-key>" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Write a haiku"}],"max_tokens":128}'
```

## Project layout

* `app/main.py` — main FastAPI app
* `app/utils.py` — mapping utilities
* `tests/test_map.py` — unit tests for mapping logic
* Docker + docker-compose for local dev.

## Notes

* Replace `.env` secrets with a secure vault in production.
* Improve auth (JWT / OAuth) and add rate-limiting for production.

```
```

---

## requirements.txt

```
fastapi==0.95.2
uvicorn[standard]==0.22.0
httpx[http2]==0.25.0
python-dotenv==1.0.0
pytest==7.4.0
```

---

## Dockerfile

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
```

---

## docker-compose.yml

```yaml
version: '3.8'
services:
  proxy:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PROXY_API_KEY=${PROXY_API_KEY}
      - TARGET_API=${TARGET_API:-responses}
      - DEFAULT_MODEL=${DEFAULT_MODEL:-gpt-5}
    ports:
      - '8000:8000'
```

---

## .env.example

```
# Copy to .env and edit
OPENAI_API_KEY=sk-REPLACE_WITH_YOUR_KEY
PROXY_API_KEY=changeme
TARGET_API=responses
DEFAULT_MODEL=gpt-5
```

---

## app/**init**.py

```python
# package marker
```

---

## app/utils.py

```python
from typing import Any, Dict, List


def messages_to_input(messages: List[Dict[str, Any]]) -> str:
    """Convert messages list to a single input string for Responses API.

    Each message becomes a block like:
    [role]
    content
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content.strip()}\n")
    return "\n---\n".join(parts)


def map_client_to_openai(body: Dict[str, Any], target_api: str, default_model: str = "gpt-5") -> Dict[str, Any]:
    """Map client-style payload (gpt-4o style) to OpenAI target API payload.

    - `max_tokens` -> `max_output_tokens` (responses) or `max_completion_tokens` (completions)
    - `messages` -> `input` (responses) or `messages` (chat)
    - `prompt` -> `input` or `prompt`
    """
    payload: Dict[str, Any] = {}

    client_model = body.get("model")
    model_map = {
        "gpt-4o": "gpt-5",
        "gpt-4o-mini": "gpt-5-mini",
    }
    payload["model"] = model_map.get(client_model, client_model or default_model)

    if "max_tokens" in body:
        v = body.get("max_tokens")
        if target_api == "responses":
            payload["max_output_tokens"] = v
        else:
            payload["max_completion_tokens"] = v

    # pass-through common parameters
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stop", "n"):
        if key in body:
            payload[key] = body[key]

    if "messages" in body:
        if target_api == "responses":
            payload["input"] = messages_to_input(body["messages"])
        else:
            payload["messages"] = body["messages"]

    if "prompt" in body:
        if target_api == "responses":
            payload["input"] = body["prompt"]
        else:
            payload["prompt"] = body["prompt"]

    return payload
```

---

## app/main.py

```python
import os
import logging
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from utils import map_client_to_openai

# load env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "changeme")
TARGET_API = os.getenv("TARGET_API", "responses")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt-proxy")

app = FastAPI(title="GPT Params Proxy")


@app.post("/proxy")
async def proxy(request: Request, x_api_key: str = Header(None)):
    # simple auth check
    if x_api_key != PROXY_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid proxy API key")

    body = await request.json()
    client_wants_stream = bool(body.get("stream", False))

    openai_payload = map_client_to_openai(body.copy(), TARGET_API, DEFAULT_MODEL)

    if TARGET_API == "responses":
        url = f"{OPENAI_BASE}/responses"
    else:
        url = f"{OPENAI_BASE}/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # streaming path
    if client_wants_stream:
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    async with client.stream("POST", url, headers=headers, json=openai_payload) as resp:
                        if resp.status_code >= 400:
                            err_text = await resp.aread()
                            logger.error("OpenAI stream error: %s", err_text.decode(errors="ignore"))
                            yield f"error: {resp.status_code}\n\n"
                            return
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk
                except Exception as e:
                    logger.exception("stream proxy error")
                    yield f"error: {str(e)}".encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # non-streaming
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=openai_payload, headers=headers, timeout=60.0)
        content = resp.text
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": content}
        if resp.status_code >= 400:
            logger.error("OpenAI error: %s", content)
            raise HTTPException(status_code=502, detail={"openai_status": resp.status_code, "body": data})
        return JSONResponse(content=data)
```

---

## tests/test\_map.py

```python
from app.utils import map_client_to_openai


def test_map_max_tokens_to_responses():
    body = {"model": "gpt-4o", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    mapped = map_client_to_openai(body, target_api="responses", default_model="gpt-5")
    assert mapped["model"] == "gpt-5"
    assert mapped["max_output_tokens"] == 100
    assert "input" in mapped


def test_map_max_tokens_to_completions():
    body = {"max_tokens": 50, "prompt": "say hello"}
    mapped = map_client_to_openai(body, target_api="completions", default_model="gpt-5")
    assert mapped["max_completion_tokens"] == 50
    assert mapped["prompt"] == "say hello"
```

---

## .github/workflows/python-app.yml

```yaml
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest -q
```

---

## Usage notes

* You can push this folder to GitHub and enable Secrets to store `OPENAI_API_KEY` for CI if needed.
* Consider swapping the `PROXY_API_KEY` simple header check for JWT or another robust auth method in production.

---


