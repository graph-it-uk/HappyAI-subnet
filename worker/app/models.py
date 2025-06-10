from uuid import uuid4

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import UUID, MetaData
from sqlalchemy.dialects.postgresql import UUID as SA_UUID
from sqlalchemy.orm import declarative_base, relationship

from app.utils.sql_connector import engine

metadata = MetaData(schema="avocado")
Base = declarative_base(metadata=metadata)


class User(Base):
    __tablename__ = "user"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid4)
    oauth2_id = Column(String(300), nullable=True)
    created_at = Column(DateTime, nullable=False)
    name = Column(String(100), nullable=True)
    gender = Column(String(50), nullable=True)
    birth_datetime = Column(DateTime, nullable=True)
    email = Column(String(100), unique=True, index=True, nullable=True)
    password = Column(String(100), nullable=False)
    image = Column(Text, nullable=True)
    refresh_token = Column(String(400), nullable=True)

    questionnaire = relationship(
        "Questionnaire",
        backref="user",
        uselist=False,
        single_parent=True,
        passive_deletes=True,
        cascade="all, delete",
    )
    moods = relationship(
        "Mood",
        backref="user",
        passive_deletes=True,
        cascade="all, delete",
        order_by="Mood.created_at",
    )


class Verification(Base):
    __tablename__ = "verification"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False)
    type = Column(String(30), nullable=False)
    secret = Column(String(100), nullable=False)
    user_id = Column(SA_UUID(as_uuid=True), nullable=False)


class Questionnaire(Base):
    __tablename__ = "questionnaire"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False)
    data = Column(JSON, nullable=False)

    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"))


class Mood(Base):
    __tablename__ = "mood"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False)
    emotions = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)
    mood_score = Column(Float, nullable=False)
    anxiety_score = Column(Float, nullable=False)
    user_id = Column(SA_UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)


class Chat(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, unique=True, nullable=False)

    __tablename__ = 'chat'

    fk_user_id = Column(UUID(as_uuid=True), nullable=False)

    started_at = Column(DateTime, nullable=False)
    session_name = Column(String, nullable=True)
    session_tag = Column(String, nullable=True)

    llm_model = Column(String, default='core', nullable=False)

    def __repr__(self):
        return (f"<Chat(id={self.id},  fk_user_id={self.fk_user_id}, "
                f"started_at={self.started_at}, session_name={self.session_name}, session_tag={self.session_tag})>")


Base.metadata.create_all(engine)
