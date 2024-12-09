import asyncio
import typing

import nltk
import tiktoken
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")


async def hybrid_chunking(
    text: str, max_tokens: int, model: str = "gpt-3.5-turbo"
) -> typing.List[str]:
    """
    Improved function to split text into token-limited chunks using paragraphs first and
    sentences as a fallback for very large paragraphs.
    """
    encoding = tiktoken.encoding_for_model(model)
    paragraphs = text.split("\n\n")  # Split text by paragraph
    chunks = []
    current_chunk = []
    current_token_count = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        paragraph_tokens = len(encoding.encode(paragraph))

        if paragraph_tokens > max_tokens:
            # Fallback to sentence-level splitting for large paragraphs
            print("Splitting paragraph into sentences due to token overflow.")
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_tokens = len(encoding.encode(sentence))

                if current_token_count + sentence_tokens > max_tokens:
                    # Finalize current chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_token_count = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_token_count += sentence_tokens
        else:
            # Add paragraph if it fits
            if current_token_count + paragraph_tokens > max_tokens:
                # Finalize current chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_token_count = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_token_count += paragraph_tokens

    # Add the last chunk if any content remains
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


async def retry_with_backoff(
    func: typing.Callable, max_retries=3, base_delay=2, **kwargs
) -> typing.Any:
    """
    Retry a coroutine function with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return await func(**kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                delay = base_delay * (2**attempt)
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2**attempt)
    raise Exception("Max retries exceeded. Failed to complete the request.")
