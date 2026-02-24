from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from db_updated import chatbot, rec_all_thread

app = FastAPI(title="Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ThreadCreateResponse(BaseModel):
    thread_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/threads")
def list_threads() -> dict:
    threads = [str(tid) for tid in rec_all_thread()]
    return {"threads": threads}


@app.post("/threads", response_model=ThreadCreateResponse)
def create_thread() -> ThreadCreateResponse:
    return ThreadCreateResponse(thread_id=str(uuid4()))


@app.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str) -> dict:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    cleaned = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            cleaned.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            cleaned.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            continue

    return {"thread_id": thread_id, "messages": cleaned}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    chunks: list[str] = []
    for message_chunk, _ in chatbot.stream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(message_chunk, AIMessage) and isinstance(message_chunk.content, str):
            if message_chunk.content:
                chunks.append(message_chunk.content)

    answer = "".join(chunks).strip()
    return {"thread_id": thread_id, "reply": answer}
