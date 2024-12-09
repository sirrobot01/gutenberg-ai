from sqlalchemy import Column, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./books.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


class BookAnalysis(Base):
    __tablename__ = "book_analysis"
    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"))
    analysis_type = Column(
        Enum("summary", "sentiment", "key_characters", "custom", name="analysis_type")
    )
    llm_type = Column(Enum("openai", "groq", "samba", name="llm_type"))
    analysis_result = Column(Text)


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    book_id = Column(String, unique=True, index=True)
    title = Column(String)
    text = Column(Text)
    authors = Column(String)  # Comma-separated list(could be a separate table)
    languages = Column(String)  # Comma-separated list
    subjects = Column(String)  # Comma-separated list
    rights = Column(String)
    publisher = Column(String)
    issued = Column(String)
    summarized_text = Column(Text)

    analysis = relationship(
        "BookAnalysis", backref="book", cascade="all, delete-orphan"
    )


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
