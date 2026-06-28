#!/usr/bin/env python3

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

from pykokoro.generation_config import GenerationConfig
from pykokoro.onnx_backend import Kokoro
from pykokoro import build_pipeline, PipelineConfig
import soundfile as sf

import pudb

VOICES_BIN_DEFAULT = "./voices-v1.0.bin"
LANG_DEFAULT = "en-US"

def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce one or more audio files from a file of quoted strings"
    )
    parser.add_argument(
        "-s",
        "--voices",
        dest="voices_path",
        nargs="?",
        default=VOICES_BIN_DEFAULT,
        help=f"Path to the voices .bin (zip) file. Defaults to {VOICES_BIN_DEFAULT}",
    )
    parser.add_argument(
        "-c",
        "--voice",
        dest="voice_name",
        nargs="?",
        default="af_sarah",
        help=f"Name of Kokoro voice to use. Defaults to 'af_sarah'.",
    )
    parser.add_argument(
        "-l",
        "--lang",
        default=LANG_DEFAULT,
        metavar="LANG",
        help=(
            "Language-country code to use for synthesis, e.g. 'en-US', 'en-UK'. "
            "Must be two lowercase letters, a hyphen, and two uppercase letters. "
            f"Defaults to '{LANG_DEFAULT}'."
        ),
    )
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        default=".",
        metavar="INPUT_FILE",
        help="File of quoted strings to read and convert to audio. Each line should contain a single quoted string. Defaults to '.' (current directory).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=".",
        metavar="OUTPUT_DIR",
        help="Directory to write output audio files to. Defaults to '.' (current directory).",
    )
    parser.add_argument(
        "-a",
        "--hash-output",
        dest="hash_output",
        default=False,
        metavar="HASH_OUTPUT",
        help="Use SHA-256 hashed output paths to write audio files to.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _validate_lang(args.lang)
    return args


_LANG_RE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")


def _validate_lang(lang: str) -> None:
    if not _LANG_RE.match(lang):
        raise argparse.ArgumentTypeError(
            f"Invalid --lang value {lang!r}. "
            "Expected format is two lowercase letters, a hyphen, and two uppercase letters (e.g. 'en-US')."
        )


def hash_path(root: Path, digest: str, levels=(2, 2)):
    parts = []
    i = 0
    for n in levels:
        parts.append(digest[i:i+n])
        i += n
    return root.joinpath(*parts, digest)


class KokoroTTSWriter:
    """
    A class to generate audio files from text using Kokoro TTS voices. It allows you to specify the voice, language, and output file name.
    """

    def __init__(self, voices_path: str, voice: str, samplerate: int = 2400,
                 output_type: str = "Opus", output_ext: str = "ogg",
                 output_dir: str = ".", lang: str = "en-us") -> None:
        """
        Initialize the VoiceWriter with the specified voice, language, and output file name.

        :param voice: The name of the Kokoro TTS voice to use.
        :param lang: The language code for the TTS generation (default is "en-us").
        """
        self.voices_path = voices_path
        self.voice = voice
        self.lang = lang
        self.samplerate = samplerate
        self.output_dir = Path(output_dir)
        self.output_ext = output_ext
        self.output_type = output_type
        self.gen_cfg = None
        self.pipe_cfg = None
        self.kokoro = None
        self.kokoro_pipe = None
        self.samples_total = 0
        self.init_kokoro()

    def init_kokoro(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gen_cfg = GenerationConfig(lang=self.lang)
        self.pipe_cfg = PipelineConfig(voices_path=Path(self.voices_path),
                                  generation=self.gen_cfg, voice=self.voice)
        self.kokoro = Kokoro(voices_path=self.pipe_cfg.voices_path)
        self.kokoro_pipe = build_pipeline(config=self.pipe_cfg, backend=self.kokoro)

    def generate(self, text: str):
        """
        Generate audio from the provided text and save it to the specified output file.

        :param text: The text to convert to speech.
        """

        samples = self.kokoro_pipe.run(text).audio
        digest = hashlib.sha256(samples).hexdigest()
        self.samples_total += len(samples)
        return samples, digest

    def write(self, text: str, output_basename: str):
        """
        Generate audio from the provided text and save it to the specified output file.
        :param text: The text to convert to speech.
        :param output_basename: The base filename of the output audio file (path and extension automatically concatenated).
        """
        output_path = self.output_dir / f"{output_basename}.{self.output_ext}"
        samples, digest = self.generate(text)
        sf.write(output_path,
                 data=samples,
                 samplerate=self.samplerate,
                 subtype=self.output_type)
        print(f"Generating audio for voice: {self.voice}... to {output_path}")
        return output_path, digest

    def write_to_hashpath(self, text: str, file_ext: str = "ogg"):
        """
        Generate audio from the provided text and save it to the specified output file.
        :param text: The text to convert to speech.
        :param output_basename: The base filename of the output audio file (path and extension automatically concatenated).
        """
        samples, digest = self.generate(text)
        output_path = hash_path(self.output_dir, digest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaf_name = re.sub(r'\W+', '_', text)[:50]  # Sanitize filename
        output_path = f"{output_path}-{leaf_name}.{self.output_ext}"
        sf.write(output_path,
                 data=samples,
                 samplerate=self.samplerate,
                 subtype=self.output_type)
        print(f"Writing audio for voice: {self.voice}... to {output_path}")
        return output_path, digest

    def close(self):
        """
        Finalize the audio writing process. This method can be used to perform any cleanup or finalization tasks.
        """
        self.gen_cfg = None
        self.pipe_cfg = None
        self.kokoro = None
        self.kokoro_pipe = None
        print(f"Total samples written: {self.samples_total}, "
              f"total time: {self.samples_total / self.samplerate:.2f} "
              f"seconds")


def main(args):
    kokoro_writer = KokoroTTSWriter(
        voices_path=args.voices_path,
        voice=args.voice_name,
        samplerate=24000,
        output_type="Opus",
        output_ext="ogg",
        output_dir=args.output_dir,
        lang=args.lang
    )

    if not (args.input_file and os.path.isfile(args.input_file)):
        kokoro_writer.write("Hello, this is a test of the Kokoro TTS system.",
                            "test_output")

    else:
        with open(args.input_file, "r") as f:
            line: str
            for line in f:
                line = line.strip("\n\r\t \"")

                print(f"Processing line: {line}")
                # Sanitize filename
                output_basename = re.sub(r'\W+', '_', line)[:50]
                if args.hash_output:
                    kokoro_writer.write_to_hashpath(line)
                else:
                    kokoro_writer.write(line, output_basename)

    kokoro_writer.close()


if __name__ == "__main__":
    main(get_args(sys.argv[1:]))
