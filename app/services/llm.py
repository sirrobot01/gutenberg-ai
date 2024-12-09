import enum
import logging

import openai
import tiktoken
from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Book
from app.services.utils import hybrid_chunking, retry_with_backoff


class LLMType(enum.Enum):
    OPENAI = "openai"
    GROQ = "groq"
    SAMBA = "samba"


class LLM:
    @classmethod
    def create(cls, db: AsyncSession, llm_type: str, token: str = None):
        llm_type = LLMType(llm_type)
        if llm_type == LLMType.OPENAI:
            return OpenAI(db, token)
        elif llm_type == LLMType.GROQ:
            return GROQ(db, token)
        elif llm_type == LLMType.SAMBA:
            return Samba(db, token)
        raise ValueError(f"Invalid LLM type: {llm_type}")


class BaseLLM:
    def init_client(self):
        return None

    def __init__(self, db: AsyncSession, token: str = None):
        self.token = token
        self.db = db
        self.base_url = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.init_client()

    async def recursive_summarize(
        self, text: str, system_prompt: str, max_tokens: int
    ) -> str:
        """
        Recursively summarize text until it fits within the token limit.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        chunk_size = max_tokens
        print("Total Tokens:", len(encoding.encode(text)))

        # Chunk text initially
        while len(encoding.encode(text)) > max_tokens:
            print(f"Text exceeds {max_tokens} tokens. Splitting into chunks...")

            # Split the text into smaller chunks
            chunks = await hybrid_chunking(text, max_tokens=chunk_size)
            print(
                f"Split into {len(chunks)} chunks with size {chunk_size} tokens each."
            )

            combined_chunks = []
            temp_chunk = ""
            for chunk in chunks:
                if len(encoding.encode(temp_chunk + chunk)) <= max_tokens:
                    temp_chunk += "\n\n" + chunk
                else:
                    combined_chunks.append(temp_chunk.strip())
                    temp_chunk = chunk
            if temp_chunk:
                combined_chunks.append(temp_chunk.strip())

            print(f"Optimized into {len(combined_chunks)} chunks after combining.")

            # Summarize each chunk
            summaries = []
            for idx, chunk in enumerate(combined_chunks):
                print(f"Summarizing chunk {idx + 1}/{len(combined_chunks)}")
                try:
                    summary = await self.call(system_prompt, chunk, max_tokens=500)
                    summaries.append(summary)
                except Exception as e:
                    self.logger.error(f"Error summarizing chunk {idx}: {e}")
                    summaries.append("")

            # Combine the summaries
            text = "\n\n".join(summaries)

        return text

    async def summarize(self, book: Book, refresh=False) -> str:
        """
        Summarize a book using recursive summarization to handle large texts.
        """
        system_prompt = (
            "You are a helpful and expert assistant in literary analysis. "
            "You can summarize texts, determine sentiment, and identify key characters. "
            "Respond concisely and accurately."
        )

        # Use recursive summarization to handle large texts
        if refresh or not book.summarized_text:
            print("Starting recursive summarization...")
            try:
                summarized_text = await self.recursive_summarize(
                    book.text, system_prompt, max_tokens=5000
                )
                book.summarized_text = summarized_text
                await self.db.commit()
            except Exception as e:
                print(f"Error during summarization: {e}")
                return ""

        return book.summarized_text

    async def call(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError

    async def analyze(self, book: Book, analysis_type: str) -> str:
        summarized_text = await self.summarize(book)
        system_prompt = (
            "You are a helpful and expert assistant in literary analysis. "
            "You can summarize texts, determine sentiment, and identify key characters. "
            "Respond concisely and accurately. Make sure not to include any introductory "
            'phrases such as such as "Here is a summary" or "The following is a summary."'
        )
        if analysis_type == "summary":
            user_prompt = (
                f"Provide a concise summary of the following text:\n\n{summarized_text}"
            )
        elif analysis_type == "sentiment":
            user_prompt = (
                f"Analyze the sentiment of the following text:\n\n{summarized_text}"
            )
        elif analysis_type == "key_characters":
            user_prompt = f"Identify the key characters in the following text:\n\n{summarized_text}"
        else:
            user_prompt = f"Analyze the following text:\n\n{summarized_text}"

        return await self.call(system_prompt, user_prompt)


class OpenAI(BaseLLM):
    client = None

    def init_client(self):
        self.client = openai.AsyncOpenAI(api_key=self.token)

    async def call(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs["model"] = kwargs.get("model", "gpt-3.5-turbo")
        kwargs["temperature"] = kwargs.get("temperature", 0.7)
        kwargs["max_tokens"] = kwargs.get("max_tokens", 100)
        kwargs["top_p"] = kwargs.get("top_p", 1.0)
        kwargs["frequency_penalty"] = kwargs.get("frequency_penalty", 0.0)
        kwargs["presence_penalty"] = kwargs.get("presence_penalty", 0.0)
        response = await retry_with_backoff(
            self.client.chat.completions.create, messages=messages, **kwargs
        )
        return response.choices[0].message["content"].strip()


class GROQ(BaseLLM):
    client = None

    def init_client(self):
        self.client = AsyncGroq(api_key=self.token)

    async def call(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs["model"] = kwargs.get("model", "llama3-8b-8192")
        kwargs["max_tokens"] = kwargs.get("max_tokens", 100)

        chat_completion = await retry_with_backoff(
            self.client.chat.completions.create, messages=messages, **kwargs
        )
        return chat_completion.choices[0].message.content.strip()


class Samba(BaseLLM):
    pass
