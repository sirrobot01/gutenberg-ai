import asyncio
import re
import typing

import httpx
import rdflib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Book


def parse(title: str, author: str, raw: str) -> str:
    """
    Extracts the main text of a book from Project Gutenberg raw text.

    :param title: Title of the book.
    :param author: Author of the book.
    :param raw: Raw text content of the book.
    :return: The extracted main text content.
    """

    # Find the start of the text
    start_regex = r"\*\*\*\s?START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    start_match = re.search(start_regex, raw, re.IGNORECASE)

    if not start_match:
        print("Could not find the start marker in the raw text.")
        return raw

    start_pos = start_match.end()

    # Refine start position based on title and author
    text_lower = raw[start_pos:].lower()
    title_match = re.search(re.escape(title.lower()), text_lower)
    if title_match:
        start_pos += title_match.end()
        author_match = re.search(
            re.escape(author.lower()), text_lower[title_match.end() :]
        )
        if author_match:
            start_pos += author_match.end()

    # Find the end of the text
    end_regex = r"end of th(?:is|e) project gutenberg ebook"
    end_match = re.search(end_regex, raw[start_pos:].lower())

    if not end_match:
        print("Could not find the end marker in the raw text.")
        return raw

    end_pos = start_pos + end_match.start()

    # Extract and return the cleaned text
    text = raw[start_pos:end_pos].strip()
    return text


class Gutenberg:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.base_url = "https://www.gutenberg.org"
        self.client = httpx.AsyncClient(follow_redirects=True)

    async def close(self) -> None:
        await self.client.aclose()

    async def get_book(self, book_id) -> typing.Optional[Book]:
        book = await self.fetch_book_db(book_id)
        if book:
            return book
        book = await self.fetch_book_api(book_id)
        return book

    async def fetch_book_api(self, book_id) -> typing.Optional[Book]:
        text_url = f"{self.base_url}/files/{book_id}/{book_id}-0.txt"
        response_task = asyncio.create_task(self.client.get(text_url))
        metadata_task = asyncio.create_task(self.extract_metadata(book_id))
        response, metadata = await asyncio.gather(response_task, metadata_task)
        if response.status_code == 200:
            book_text = response.text
            book_text = parse(metadata["title"], metadata["authors"], book_text)
            book_text = book_text.strip()
            book = await self.save_book_to_db(book_id, book_text, metadata)
            return book
        else:
            return None

    @staticmethod
    def _get_single_value(
        g: rdflib.Graph, subject: rdflib.URIRef, predicate: rdflib.URIRef
    ) -> typing.Optional[str]:
        val = next(g.objects(subject, predicate), None)
        return str(val) if val is not None else None

    async def extract_metadata(self, book_id) -> typing.Dict[str, typing.Any]:
        url = f"{self.base_url}/ebooks/{book_id}.rdf"
        data = {
            "title": None,
            "authors": "",
            "publisher": None,
            "issued": None,
            "languages": "",
            "subjects": "",
            "rights": None,
        }
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code == 200:
                rdf_content = response.text
                g = rdflib.Graph()
                DC = rdflib.Namespace("http://purl.org/dc/terms/")
                PGTERMS = rdflib.Namespace("http://www.gutenberg.org/2009/pgterms/")
                RDF = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
                book_uri = rdflib.URIRef(f"http://www.gutenberg.org/ebooks/{book_id}")
                g.parse(data=rdf_content, format="xml")

                data["publisher"] = self._get_single_value(g, book_uri, DC.publisher)
                data["issued"] = self._get_single_value(g, book_uri, DC.issued)
                data["rights"] = self._get_single_value(g, book_uri, DC.rights)
                data["title"] = self._get_single_value(g, book_uri, DC.title)

                authors = set()
                for creator in g.objects(book_uri, DC.creator):
                    for name in g.objects(creator, PGTERMS.name):
                        authors.add(str(name))

                languages = set()
                for lang_node in g.objects(book_uri, DC.language):
                    for lang_val in g.objects(lang_node, RDF.value):
                        languages.add(str(lang_val))

                subjects = set()
                for subject_node in g.objects(book_uri, DC.subject):
                    for val in g.objects(subject_node, RDF.value):
                        subjects.add(str(val))

                data["authors"] = " | ".join(authors)
                data["languages"] = ", ".join(languages)
                data["subjects"] = ", ".join(subjects)
        return data

    async def fetch_book_db(self, book_id) -> typing.Optional[Book]:
        query = select(Book).where(Book.book_id == book_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def save_book_to_db(
        self, book_id, text: str, metadata: typing.Dict[str, typing.Any]
    ) -> Book:
        book = Book(book_id=book_id, text=text)
        book.title = metadata["title"]
        book.authors = metadata["authors"]
        book.publisher = metadata["publisher"]
        book.issued = metadata["issued"]
        book.languages = metadata["languages"]
        book.subjects = metadata["subjects"]
        book.rights = metadata["rights"]
        self.db.add(book)
        await self.db.commit()
        return book
