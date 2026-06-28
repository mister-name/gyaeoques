#!/usr/bin/env python3
"""
Embedding generator for jokes.

Queries joke rows where embed IS NULL, generates a 1024-dimensional semantic
embedding vector using the Qwen3-Embedding-0.6B model via the Transformers
library, and updates each row with the resulting vector.

Reference:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from sqlalchemy import select
from transformers import AutoModel, AutoTokenizer

from db.database import get_session, init_db
from db.models import Joke

EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMB_DIMS = 1024


# ---------------------------------------------------------------------------
# Embedding helpers  (adapted from HuggingFace Transformers Usage example)
# ---------------------------------------------------------------------------

def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last non-padding token representation for each sequence."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# ---------------------------------------------------------------------------
# Embedding generator class
# ---------------------------------------------------------------------------

class JokeEmbeddingGenerator:
    """
    Loads the Qwen3-Embedding model once and generates embeddings on demand.
    """

    TASK = "Generate a semantic embedding for the following joke text"

    def __init__(self, model_name: str = EMB_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        # Default to CPU: torch.cuda.is_available() can return True even when
        # the GPU's compute capability is not supported by the installed PyTorch
        # build (e.g. Pascal / sm_61 with PyTorch >=2.x).  Pass device="cuda"
        # explicitly only if you know your GPU is supported.
        self.device = device or "cpu"
        print(f"Loading tokenizer and model '{model_name}' on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        Return a list of normalised embedding vectors (one per input text).
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [
                _get_detailed_instruct(self.TASK, t)
                for t in texts[i: i + batch_size]
            ]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            embeddings = _last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def generate_embeddings(model_name: str = EMB_MODEL, batch_size: int = 8, device: str | None = None) -> None:
    """
    Iterate all joke rows where embed IS NULL, generate embeddings, and
    persist the results back to the database.
    """
    generator = JokeEmbeddingGenerator(model_name=model_name, device=device)

    with get_session() as session:
        stmt = select(Joke).where(Joke.embed.is_(None))
        jokes = session.scalars(stmt).all()
        print(f"Found {len(jokes)} joke(s) without embeddings.")

        for i in range(0, len(jokes), batch_size):
            batch = jokes[i: i + batch_size]
            texts = [j.content for j in batch]
            embeddings = generator.embed(texts, batch_size=batch_size)

            for joke, vector in zip(batch, embeddings):
                joke.embed = vector

            session.commit()
            print(f"Committed embeddings for jokes {i + 1}–{i + len(batch)}.")

    print("Done generating embeddings.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic embeddings for all jokes that lack one, "
            f"using the {EMB_MODEL} model."
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
        "--model",
        dest="model_name",
        default=EMB_MODEL,
        metavar="MODEL",
        help=f"HuggingFace model identifier (default: {EMB_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        metavar="N",
        help="Number of jokes to embed per forward pass (default: 8).",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        metavar="DEVICE",
        help=(
            "Torch device to run inference on, e.g. 'cpu' or 'cuda'. "
            "Defaults to 'cpu'.  Use 'cuda' only if your GPU's compute "
            "capability is supported by the installed PyTorch build."
        ),
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    init_db(args.db_url)
    generate_embeddings(model_name=args.model_name, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
