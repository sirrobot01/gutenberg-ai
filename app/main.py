import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic.v1 import BaseSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Book, get_db, init_db
from app.services import Services


class Settings(BaseSettings):
    LLM_TYPE: str = "openai"
    LLM_TOKEN: str
    REQUEST_PER_MINUTE: int = 10

    class Config:
        # Get env file from base directory
        env_file = Path(__file__).parent.parent / ".env"
        frozen = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_services(
    db: AsyncSession = Depends(get_db), settings: Settings = Depends(get_settings)
) -> Services:
    return Services(db, settings.LLM_TYPE, settings.LLM_TOKEN)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

rate_limit_store = defaultdict(list)


async def rate_limit(request: Request):
    settings = get_settings()
    # Get client IP
    client_ip = request.client.host
    now = time.time()

    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip] if now - req_time < 60
    ]

    # Check if too many requests
    if len(rate_limit_store[client_ip]) >= settings.REQUEST_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests")

    # Add current request
    rate_limit_store[client_ip].append(now)
    return True


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/fetch_book", response_class=HTMLResponse)
async def fetch_book(
    request: Request,
    book_id: str = Form(...),
    services: Services = Depends(get_services),
):
    book = await services.gutenberg.get_book(book_id)
    if book:
        await services.gutenberg.close()
        return templates.TemplateResponse(
            "fragments/book_detail.html", {"request": request, "book": book}
        )
    else:
        return HTMLResponse("<p>Book not found or an error occurred.</p>")


@app.post("/analyze/{book_id}", response_class=HTMLResponse)
async def analyze(
    request: Request,
    book_id: str,
    analysis_type: str = Form(...),
    services: Services = Depends(get_services),
):
    await rate_limit(request)
    book = await services.gutenberg.get_book(book_id)
    if book:
        analysis_result = await services.llm.analyze(book, analysis_type)
        return f"<div class='text-gray-700 text-lg'>{analysis_result}</div>"
    else:
        return HTMLResponse("<p>Book not found.</p>")


@app.get("/books", response_class=HTMLResponse)
async def books_list(request: Request, db: AsyncSession = Depends(get_db)):
    books = await db.execute(select(Book))
    books = books.scalars().all()
    return templates.TemplateResponse(
        "books.html", {"books": books, "request": request}
    )


@app.get("/book/{book_id}", response_class=HTMLResponse)
async def book_detail(
    request: Request, book_id: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Book).where(Book.book_id == book_id))
    book = result.scalar_one_or_none()
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    return templates.TemplateResponse(
        "book_detail.html", {"request": request, "book": book}
    )
