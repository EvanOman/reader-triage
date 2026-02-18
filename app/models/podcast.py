"""SQLAlchemy models for podcasts, episodes, scores, and tags."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.article import Base


class Podcast(Base):
    """A podcast subscription imported from OPML."""

    __tablename__ = "podcasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500))
    feed_url: Mapped[str] = mapped_column(String(2000), unique=True)
    youtube_channel_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    youtube_channel_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    mapping_confirmed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    episodes: Mapped[list["PodcastEpisode"]] = relationship(
        back_populates="podcast", cascade="all, delete-orphan"
    )


class PodcastEpisode(Base):
    """An individual podcast episode."""

    __tablename__ = "podcast_episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    podcast_id: Mapped[int] = mapped_column(Integer, ForeignKey("podcasts.id"))
    guid: Mapped[str] = mapped_column(String(500), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    audio_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_source: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # youtube / groq_whisper
    youtube_video_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / transcript_ready / scored / failed
    listened: Mapped[bool] = mapped_column(default=False)

    # Relationships
    podcast: Mapped["Podcast"] = relationship(back_populates="episodes")
    score: Mapped["PodcastEpisodeScore | None"] = relationship(
        back_populates="episode", uselist=False
    )
    tags: Mapped[list["PodcastEpisodeTag"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_episodes_podcast", "podcast_id"),
        Index("idx_episodes_status", "status"),
        Index("idx_episodes_published", "published_at"),
    )


class PodcastEpisodeScore(Base):
    """AI-generated score for a podcast episode."""

    __tablename__ = "podcast_episode_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[int] = mapped_column(Integer, ForeignKey("podcast_episodes.id"), unique=True)

    # Score components (0-25 each, total 0-100) - same schema as ArticleScore
    specificity_score: Mapped[int] = mapped_column(Integer, default=0)
    novelty_score: Mapped[int] = mapped_column(Integer, default=0)
    depth_score: Mapped[int] = mapped_column(Integer, default=0)
    actionability_score: Mapped[int] = mapped_column(Integer, default=0)

    # Explanations
    score_reasons: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    overall_assessment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scoring_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    episode: Mapped["PodcastEpisode"] = relationship(back_populates="score")

    @property
    def info_score(self) -> int:
        """Total score (0-100), matching ArticleScore.info_score semantics."""
        return (
            self.specificity_score
            + self.novelty_score
            + self.depth_score
            + self.actionability_score
        )

    @classmethod
    def score_total_expr(cls):
        """SQLAlchemy column expression for the total score.

        Use in queries: .where(PodcastEpisodeScore.score_total_expr() >= 60)
        """
        return cls.specificity_score + cls.novelty_score + cls.depth_score + cls.actionability_score

    __table_args__ = (Index("idx_episode_scores_episode", "episode_id"),)


class PodcastEpisodeTag(Base):
    """Association between a podcast episode and a topic tag."""

    __tablename__ = "podcast_episode_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[int] = mapped_column(Integer, ForeignKey("podcast_episodes.id"))
    tag_slug: Mapped[str] = mapped_column(String(50))
    tagged_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    tagging_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    episode: Mapped["PodcastEpisode"] = relationship(back_populates="tags")

    __table_args__ = (
        Index("idx_episode_tags_episode", "episode_id"),
        Index("idx_episode_tags_tag", "tag_slug"),
    )
