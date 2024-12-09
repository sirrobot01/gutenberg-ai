from sqlalchemy.ext.asyncio import AsyncSession

from .gutenberg import Gutenberg
from .llm import LLM


class Services:
    def __init__(self, db: AsyncSession, llm_type: str = None, llm_token: str = None):
        self.db = db
        self.llm_type = llm_type
        self.llm_token = llm_token

    @property
    def gutenberg(self):
        return Gutenberg(self.db)

    @property
    def llm(self):
        return LLM.create(self.db, self.llm_type, self.llm_token)
