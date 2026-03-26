from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns that may be missing on older databases
        for stmt in [
            "ALTER TABLE annotations ADD COLUMN free_text TEXT",
            "ALTER TABLE annotations ADD COLUMN action_item_quality VARCHAR(40)",
        ]:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # Column already exists


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
