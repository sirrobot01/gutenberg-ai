from unittest.mock import Mock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.database import Book
from app.services.gutenberg import Gutenberg


@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Book.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Book.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture
def sample_metadata():
    return {
        "title": "Sample Book",
        "authors": "John Doe",
        "publisher": "Project Gutenberg",
        "issued": "2020-01-01",
        "languages": "en",
        "subjects": "Fiction",
        "rights": "Public domain",
    }


@pytest.fixture
def sample_rdf_content():
    return """<?xml version="1.0" encoding="utf-8"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/terms/"
             xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/">
        <pgterms:ebook rdf:about="http://www.gutenberg.org/ebooks/1234">
            <dc:title>Sample Book</dc:title>
            <dc:creator>
                <pgterms:agent>
                    <pgterms:name>John Doe</pgterms:name>
                </pgterms:agent>
            </dc:creator>
            <dc:publisher>Project Gutenberg</dc:publisher>
            <dc:issued>2020-01-01</dc:issued>
            <dc:language>
                <rdf:Description>
                    <rdf:value>en</rdf:value>
                </rdf:Description>
            </dc:language>
            <dc:subject>
                <rdf:Description>
                    <rdf:value>Fiction</rdf:value>
                </rdf:Description>
            </dc:subject>
            <dc:rights>Public domain</dc:rights>
        </pgterms:ebook>
    </rdf:RDF>
    """


@pytest.mark.asyncio
async def test_get_book_from_db(db_session, sample_metadata):
    # Arrange
    book_id = 1234
    gutenberg = Gutenberg(db_session)
    existing_book = Book(book_id=book_id, text="Sample text", **sample_metadata)
    db_session.add(existing_book)
    await db_session.commit()

    # Act
    result = await gutenberg.get_book(book_id)

    # Assert
    assert result is not None
    assert result.book_id == book_id
    assert result.text == "Sample text"
    assert result.title == sample_metadata["title"]
    await gutenberg.close()


@pytest.mark.asyncio
async def test_get_book_from_api(db_session, sample_rdf_content):
    # Arrange
    book_id = 1234
    gutenberg = Gutenberg(db_session)

    # Mock HTTP responses
    text_response = Mock(spec=httpx.Response)
    text_response.status_code = 200
    text_response.text = (
        "*** START OF THE PROJECT ***\nBook content\n*** END OF THE PROJECT ***"
    )

    rdf_response = Mock(spec=httpx.Response)
    rdf_response.status_code = 200
    rdf_response.text = sample_rdf_content

    async def mock_get(url):
        if url.endswith(".txt"):
            return text_response
        elif url.endswith(".rdf"):
            return rdf_response
        raise ValueError(f"Unexpected URL: {url}")

    with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
        # Act
        result = await gutenberg.get_book(book_id)

    # Assert
    assert result is not None
    assert result.book_id == book_id
    assert result.title == "Sample Book"
    assert result.authors == "John Doe"
    await gutenberg.close()


@pytest.mark.asyncio
async def test_get_book_not_found(db_session):
    # Arrange
    book_id = 9999
    gutenberg = Gutenberg(db_session)

    # Mock HTTP response for non-existent book
    response = Mock(spec=httpx.Response)
    response.status_code = 404

    with patch.object(httpx.AsyncClient, "get", return_value=response):
        # Act
        result = await gutenberg.get_book(book_id)

    # Assert
    assert result is None
    await gutenberg.close()


@pytest.mark.asyncio
async def test_extract_metadata(db_session, sample_rdf_content):
    # Arrange
    book_id = 1234
    gutenberg = Gutenberg(db_session)

    # Mock HTTP response
    response = Mock(spec=httpx.Response)
    response.status_code = 200
    response.text = sample_rdf_content

    with patch.object(httpx.AsyncClient, "get", return_value=response):
        # Act
        metadata = await gutenberg.extract_metadata(book_id)

    # Assert
    assert metadata["title"] == "Sample Book"
    assert metadata["authors"] == "John Doe"
    assert metadata["publisher"] == "Project Gutenberg"
    assert metadata["issued"] == "2020-01-01"
    assert metadata["languages"] == "en"
    assert metadata["subjects"] == "Fiction"
    assert metadata["rights"] == "Public domain"
    await gutenberg.close()


@pytest.mark.asyncio
async def test_save_book_to_db(db_session, sample_metadata):
    # Arrange
    book_id = 1234
    text = "Sample book text"
    gutenberg = Gutenberg(db_session)

    # Act
    book = await gutenberg.save_book_to_db(book_id, text, sample_metadata)

    # Assert
    assert book.book_id == book_id
    assert book.text == text
    assert book.title == sample_metadata["title"]
    assert book.authors == sample_metadata["authors"]
    assert book.publisher == sample_metadata["publisher"]
    assert book.issued == sample_metadata["issued"]
    assert book.languages == sample_metadata["languages"]
    assert book.subjects == sample_metadata["subjects"]
    assert book.rights == sample_metadata["rights"]

    # Verify it was actually saved to DB
    saved_book = await gutenberg.fetch_book_db(book_id)
    assert saved_book is not None
    assert saved_book.book_id == book_id
    await gutenberg.close()
