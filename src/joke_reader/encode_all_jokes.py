#!/usr/bin/env python3
"""
Kokoro TTS encoder for all un-encoded jokes.

Queries the database for jokes that have no corresponding media row yet
(LEFT OUTER JOIN where media.joke_uuid IS NULL), then generates an audio
file for each via KokoroTTSWriter and records the result in the media table.
"""

import argparse
import sys
import uuid

from sqlalchemy import select, or_, and_
from sqlalchemy.orm import joinedload

from db.database import get_session, init_db
from db.models import Joke, Media
from joke_reader.encode_jokes import KokoroTTSWriter


VOICES_BIN_DEFAULT = "~/projs/speak/voices-v1.0.bin"
VOICE_DEFAULT = "af_sarah"
LANG_DEFAULT = "en-US"
OUTPUT_DIR_DEFAULT = "./media"


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Kokoro TTS audio files for all jokes that do not yet "
            "have a corresponding media entry."
        )
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        metavar="DATABASE_URL",
        help=(
            "SQLAlchemy database URL.  "
            "Falls back to the DATABASE_URL environment variable."
        ),
    )
    parser.add_argument(
        "-s", "--voices",
        dest="voices_path",
        default=VOICES_BIN_DEFAULT,
        metavar="VOICES_BIN",
        help=f"Path to the Kokoro voices .bin file (default: {VOICES_BIN_DEFAULT}).",
    )
    parser.add_argument(
        "-c", "--voice",
        dest="voice_name",
        default=VOICE_DEFAULT,
        metavar="VOICE",
        help=f"Kokoro voice name (default: {VOICE_DEFAULT}).",
    )
    parser.add_argument(
        "-l", "--lang",
        dest="lang",
        default=LANG_DEFAULT,
        metavar="LANG",
        help=f"Language-country code (default: {LANG_DEFAULT}).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        default=OUTPUT_DIR_DEFAULT,
        metavar="OUTPUT_DIR",
        help=f"Directory for generated audio files (default: {OUTPUT_DIR_DEFAULT}).",
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def encode_all_jokes(
    voices_path: str,
    voice_name: str,
    lang: str,
    output_dir: str,
) -> None:
    """
    Find all joke rows without a media entry and generate a Kokoro TTS audio
    file for each.

    For each qualifying joke:
    1. Insert a placeholder media row (hash=NULL, path=NULL) to signal that
       encoding is in progress.
    2. Encode the joke text with KokoroTTSWriter.write_to_hashpath().
    3. Update the media row with the resulting path and digest.
    """
    kokoro_writer = KokoroTTSWriter(
        voices_path=voices_path,
        voice=voice_name,
        samplerate=24000,
        output_type="Opus",
        output_ext="ogg",
        output_dir=output_dir,
        lang=lang,
    )

    try:
        with get_session() as session:
            # SELECT joke rows that have no media entry yet.
            stmt = (
                select(Joke)
                .outerjoin(Media, Media.joke_uuid == Joke.uuid)
                .where(or_(Media.joke_uuid.is_(None),
                       Media.hash.is_(None),
                       Media.path.is_(None)))
                .options(joinedload(Joke.media))
            )
            jokes = session.scalars(stmt).unique().all()
            print(f"Found {len(jokes)} joke(s) to encode.")

            for joke in jokes:
                # 1. Insert placeholder media row.
                media_row = Media(
                    uuid=uuid.uuid4(),
                    joke_uuid=joke.uuid,
                    hash=None,
                    path=None,
                )
                session.add(media_row)
                session.commit()

                # 2. Generate audio.
                try:
                    output_path, digest = kokoro_writer.write_to_hashpath(joke.content)
                except Exception as exc:
                    print(
                        f"Error encoding joke {joke.uuid}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # 3. Update the media row with actual path and hash.
                media_row.path = str(output_path)
                media_row.hash = digest
                session.commit()
                print(f"Encoded joke {joke.uuid} → {output_path}")
    finally:
        kokoro_writer.close()


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    init_db(args.db_url)
    encode_all_jokes(
        voices_path=args.voices_path,
        voice_name=args.voice_name,
        lang=args.lang,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
