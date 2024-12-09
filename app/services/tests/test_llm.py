from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Book
from app.services.llm import GROQ, LLM, BaseLLM, LLMType, OpenAI


@pytest.fixture
async def db_session():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    return mock_session


@pytest.fixture
def sample_book(db_session):
    return Book(
        book_id=1,
        text="This is a sample book text for testing. It contains multiple sentences and paragraphs.",
        summarized_text=None,
    )


@pytest.fixture
def mock_openai_response():
    mock = AsyncMock()
    mock.choices = [AsyncMock(message={"content": "Test summary response"})]
    return mock


@pytest.fixture
def mock_groq_response():
    mock = AsyncMock()
    mock.choices = [AsyncMock(message=AsyncMock(content="Test summary response"))]
    return mock


@pytest.mark.asyncio
async def test_llm_factory_creation():
    # Test OpenAI creation
    llm = LLM.create(Mock(spec=AsyncSession), "openai", "test-token")
    assert isinstance(llm, OpenAI)

    # Test GROQ creation
    llm = LLM.create(Mock(spec=AsyncSession), "groq", "test-token")
    assert isinstance(llm, GROQ)

    # Test invalid type
    with pytest.raises(ValueError):
        LLM.create(Mock(spec=AsyncSession), "invalid", "test-token")


@pytest.mark.asyncio
async def test_openai_call(db_session, mock_openai_response):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
    ) as mock_retry:
        mock_retry.return_value = mock_openai_response

        llm = OpenAI(db_session, "test-token")
        result = await llm.call("System prompt", "User prompt", model="gpt-3.5-turbo")

        assert result == "Test summary response"
        mock_retry.assert_called_once()


@pytest.mark.asyncio
async def test_groq_call(db_session, mock_groq_response):
    mock_chat = Mock()
    mock_chat.completions = AsyncMock()
    mock_chat.completions.create.return_value = mock_groq_response

    mock_client = Mock()
    mock_client.chat = mock_chat

    # Try patching with the import path as used in the GROQ class file
    with (
        patch("app.services.llm.AsyncGroq", return_value=mock_client),
        patch(
            "app.services.llm.retry_with_backoff", return_value=mock_groq_response
        ) as mock_retry,
    ):
        llm = GROQ(db_session, "test-token")
        llm.init_client()

        result = await llm.call("System prompt", "User prompt", model="llama3-8b-8192")

        assert result == "Test summary response"
        mock_retry.assert_called_once_with(
            mock_client.chat.completions.create,
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User prompt"},
            ],
            model="llama3-8b-8192",
            max_tokens=100,
        )


@pytest.mark.asyncio
async def test_analyze_with_different_types(db_session, sample_book):
    class TestLLM(BaseLLM):
        async def call(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
            return f"Analysis for: {user_prompt[:20]}..."

        async def summarize(self, book: Book) -> str:
            return "Test summarized text"

    llm = TestLLM(db_session)

    # Test summary analysis
    summary = await llm.analyze(sample_book, "summary")
    assert "Analysis for: Provide a concise" in summary

    # Test sentiment analysis
    sentiment = await llm.analyze(sample_book, "sentiment")
    assert "Analysis for: Analyze the sentim" in sentiment

    # Test character analysis
    characters = await llm.analyze(sample_book, "key_characters")
    assert "Analysis for: Identify the key c" in characters


@pytest.mark.asyncio
async def test_llm_type_enum():
    assert LLMType.OPENAI.value == "openai"
    assert LLMType.GROQ.value == "groq"
    assert LLMType.SAMBA.value == "samba"
