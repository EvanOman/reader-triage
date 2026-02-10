"""Migration script to move from v1 schema (scored_articles) to v2 schema (articles + article_scores).

Run with: uv run python -m app.migrations.migrate_v2
"""

import sqlite3
from pathlib import Path


def migrate():
    """Migrate the database from v1 to v2 schema."""
    db_path = Path(__file__).parent.parent.parent / "reader_triage.db"

    if not db_path.exists():
        print("No database found, nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if old table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scored_articles'")
    if not cursor.fetchone():
        print("No scored_articles table found, nothing to migrate.")
        conn.close()
        return

    # Check if new tables already exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    if cursor.fetchone():
        print("New schema already exists. Skipping migration.")
        conn.close()
        return

    print("Migrating from v1 (scored_articles) to v2 (articles + article_scores)...")

    # Create new tables
    cursor.executescript("""
    -- Create new articles table
    CREATE TABLE articles (
        id VARCHAR(50) NOT NULL PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        url VARCHAR(2000) NOT NULL,
        author VARCHAR(200),
        word_count INTEGER,
        content_preview TEXT,
        location VARCHAR(20),
        category VARCHAR(20),
        site_name VARCHAR(200),
        reading_progress FLOAT,
        readwise_created_at DATETIME,
        readwise_updated_at DATETIME,
        first_synced_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        last_synced_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
    );

    -- Create article_scores table
    CREATE TABLE article_scores (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        article_id VARCHAR(50) NOT NULL UNIQUE,
        info_score FLOAT DEFAULT 0.0 NOT NULL,
        specificity_score INTEGER DEFAULT 0 NOT NULL,
        novelty_score INTEGER DEFAULT 0 NOT NULL,
        depth_score INTEGER DEFAULT 0 NOT NULL,
        actionability_score INTEGER DEFAULT 0 NOT NULL,
        score_reasons TEXT DEFAULT '[]' NOT NULL,
        overall_assessment TEXT,
        priority_score FLOAT,
        author_boost FLOAT DEFAULT 0.0 NOT NULL,
        priority_signals TEXT,
        skip_recommended BOOLEAN DEFAULT 0 NOT NULL,
        skip_reason VARCHAR(500),
        model_used VARCHAR(50),
        scored_at DATETIME,
        priority_computed_at DATETIME,
        FOREIGN KEY(article_id) REFERENCES articles (id)
    );

    -- Create authors table
    CREATE TABLE authors (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(200) NOT NULL UNIQUE,
        normalized_name VARCHAR(200) NOT NULL,
        total_highlights INTEGER DEFAULT 0 NOT NULL,
        total_books INTEGER DEFAULT 0 NOT NULL,
        first_highlighted_at DATETIME,
        last_highlighted_at DATETIME,
        is_favorite BOOLEAN DEFAULT 0 NOT NULL,
        notes TEXT,
        last_synced_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
    );

    -- Create author_books table
    CREATE TABLE author_books (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        author_id INTEGER NOT NULL,
        readwise_book_id INTEGER NOT NULL UNIQUE,
        title VARCHAR(500) NOT NULL,
        category VARCHAR(50),
        source_url VARCHAR(2000),
        cover_image_url VARCHAR(2000),
        num_highlights INTEGER DEFAULT 0 NOT NULL,
        last_highlight_at DATETIME,
        FOREIGN KEY(author_id) REFERENCES authors (id)
    );

    -- Create indexes
    CREATE INDEX idx_articles_location ON articles (location);
    CREATE INDEX idx_articles_author ON articles (author);
    CREATE INDEX idx_scores_priority ON article_scores (priority_score);
    CREATE INDEX idx_scores_info ON article_scores (info_score);
    CREATE INDEX idx_authors_normalized ON authors (normalized_name);
    CREATE INDEX idx_authors_highlights ON authors (total_highlights);
    CREATE INDEX idx_author_books_author ON author_books (author_id);
    """)

    # Migrate data from scored_articles to articles and article_scores
    cursor.execute("""
    INSERT INTO articles (id, title, url, author, word_count, content_preview, first_synced_at, last_synced_at)
    SELECT id, title, url, author, word_count, content_preview, created_at, created_at
    FROM scored_articles
    """)

    cursor.execute("""
    INSERT INTO article_scores (article_id, info_score, specificity_score, novelty_score, depth_score,
                                actionability_score, score_reasons, overall_assessment, scored_at)
    SELECT id, info_score, specificity_score, novelty_score, depth_score,
           actionability_score, score_reasons, overall_assessment, scored_at
    FROM scored_articles
    """)

    # Update summaries foreign key (it already references the correct id column)
    # Just need to ensure the FK constraint will work with new table

    # Rename old table as backup
    cursor.execute("ALTER TABLE scored_articles RENAME TO scored_articles_backup_v1")

    conn.commit()

    # Verify migration
    cursor.execute("SELECT COUNT(*) FROM articles")
    article_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM article_scores")
    score_count = cursor.fetchone()[0]

    print("Migration complete!")
    print(f"  - Migrated {article_count} articles")
    print(f"  - Migrated {score_count} scores")
    print("  - Old table backed up as 'scored_articles_backup_v1'")

    conn.close()


if __name__ == "__main__":
    migrate()
