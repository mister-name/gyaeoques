#!/usr/bin/env python3
"""
Joke ingestion module.

Provides a Python functional interface for adding new joke texts to the
database, plus a CLI argparse interface that lets the user specify a file of
jokes to load or supply a joke directly on the command line.

Usage:
# Ingest a file of jokes (one per line):
    python -m joke_reader.ingester -f jokes.txt

# Ingest a single joke from the command line:
    python -m joke_reader.ingester "Why did the scarecrow win an award? He was outstanding in his field."
"""

import argparse
import sys
import uuid

from db.database import get_session, init_db
from db.models import Joke


################
# Core functions
################
def insert_joke(text: str) -> Joke:
    """
    Insert a single joke into the database after sanitising its text.

    Leading/trailing whitespace and quotation characters (' " ` \u2018 \u2019
    \u201c \u201d) are stripped before insertion.

    :param text: The raw joke text.
    :returns: The newly created :class:`~db.models.Joke` ORM instance.
    """
    sanitised = text.strip(" \t\n\r'\"`\u2018\u2019\u201c\u201d")
    if not sanitised:
        return None

    with get_session() as session:
        joke = Joke(uuid=uuid.uuid4(), content=sanitised)
        session.add(joke)
        session.commit()
        session.refresh(joke)
        print(f"Inserted joke {joke.uuid}: {sanitised[:60]!r}")
        return joke


def insert_jokes(filename: str) -> list[Joke]:
    """
    Read jokes from *filename* (one per line) and insert each into the
    database via :func:`insert_joke`.

    :param filename: Path to a plain-text file containing one joke per line.
    :returns: A list of the inserted :class:`~db.models.Joke` instances
        (``None`` entries for blank/whitespace-only lines are excluded).
    """
    jokes: list[Joke] = []
    with open(filename, "r", encoding="utf-8") as fh:
        for line in fh:
            result = insert_joke(line)
            if result is not None:
                jokes.append(result)
    print(f"Inserted {len(jokes)} joke(s) from {filename!r}.")
    return jokes

##########
# CLI args
##########
def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Build and return the parsed argument namespace.

    :param argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="Ingest jokes into the joke database.",
        epilog=(
            "Exactly one of --file or a positional JOKE argument must be "
            "provided."
        ),
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="pathname",
        metavar="PATHNAME",
        default=None,
        help="Path to a text file of jokes (one per line) to ingest.",
    )
    parser.add_argument(
        "joke",
        nargs="?",
        default=None,
        metavar="JOKE",
        help="A single joke text string to ingest.",
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        metavar="DATABASE_URL",
        help=(
            "SQLAlchemy database URL "
            "(e.g. postgresql+psycopg2://user:pw@host/db).  "
            "Falls back to the DATABASE_URL environment variable."
        ),
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    init_db(args.db_url)

    if args.pathname:
        insert_jokes(args.pathname)
    elif args.joke:
        insert_joke(args.joke)
    else:
        print(
            "Error: provide either --file PATHNAME or a positional JOKE argument.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
