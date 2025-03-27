from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app import chat

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app")


@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse("chat_template.html", {"request": request})

app.include_router(chat.router)
