"""
Examines OpenLibrary tags

USAGE:
    # First, create a handy bash alias
    > alias tags='just run'

    # Examine the default file - works.txt
    > tags

    # Use a filename other than the default "works.txt"
    > tags --filename something-else.txt

    # Downloads the latest tag data, saves the file into the current directory as "works.txt", and examines it
    > tags --download

    # Examines the first 1,000,000 lines of the file only (useful for quick testing)
    > tags --limit 1000000

"""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import gzip
import shutil
import sys
import time
from typing import Optional
import unicodedata

import orjson
import requests


#
# GLOBALS
#
DEBUG = False
DEBUG_OUTPUT_QUANTA = 10_000
DEFAULT_LINE_LIMIT = 0
DEFAULT_FILENAME = "works.txt"
DEFAULT_WORKS_URL = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"


#
# TYPES
#
@contextmanager
def Timer(name: str):
    start_time = time.time()
    time.sleep(1.5)
    end_time = time.time()

    try:
        print(f"{name} ...")
        yield

    finally:
        end_time = time.time()
        diff_seconds = end_time - start_time
        print(f"{name} took {diff_seconds:.1f}s")

        # MS_PER_SEC = 1000
        # diff_ms = round(diff_seconds * MS_PER_SEC)
        # print(f"{name} took {diff_ms:,} ms")


# @dataclass
# class AuthorRole:
#    type: dict
#    author: dict


@dataclass
class Work:
    type: str
    id: str
    last_modified: str

    subjects: list[str] = field(default_factory=list)
    subject_places: list[str] = field(default_factory=list)
    subject_times: list[str] = field(default_factory=list)
    subject_people: list[str] = field(default_factory=list)

    # revision: int
    # title: Optional[str] = None
    # subtitle: Optional[str] = None
    # authors: list[AuthorRole] = field(default_factory=list)
    # translated_titles: list[str] = field(default_factory=list)
    # description: Optional[str] = None
    # dewey_number: list[str] = field(default_factory=list)
    # lc_classifications: list[str] = field(default_factory=list)
    # first_sentence: Optional[str] = None
    # original_languages: list[str] = field(default_factory=list)
    # other_titles: list[str] = field(default_factory=list)
    # first_publish_date: Optional[str] = None
    # links: list[dict] = field(default_factory=list)
    # notes: Optional[str] = None
    # cover_edition: Optional[dict] = None
    # covers: list[int] = field(default_factory=list)
    # latest_revision: Optional[int] = None
    # created: Optional[dict] = None
    # key: Optional[str] = None



# FUNCTIONS
#
def main():
    """
    This is main
    """
    args = get_args()
    set_debug(args.debug)

    if args.download:
        ok = download(file_path=args.filename, line_limit=args.limit)
        if not ok:
            return

    works = parse_works_file(file_path=args.filename, line_limit=args.limit)
    analyze_tags(works)


def analyze_tags(works: list[Work]):
    all_tags = []
    for work in works:
        all_tags.extend(work.subjects)
        all_tags.extend(work.subject_people)
        all_tags.extend(work.subject_places)
        all_tags.extend(work.subject_times)

    subject_people = [x.subject_people for x in works if x.subject_people]
    subject_places = [x.subject_places for x in works if x.subject_places]
    subject_times = [x.subject_times for x in works if x.subject_times]

    # Normalize and count
    normalized_tags = [normalize_tag(tag) for tag in all_tags]
    tag_counter = Counter(normalized_tags)

    # Find duplicates, misspellings, and capitalization differences
    duplicates = [tag for tag, count in tag_counter.items() if count > 1]
    print("Duplicate tags (case-insensitive):", duplicates)

    # Find tags differing only by capitalization
    case_variants = {}
    for tag in all_tags:
        norm = normalize_tag(tag)
        if norm not in case_variants:
            case_variants[norm] = set()
        case_variants[norm].add(tag)

    for norm, variants in case_variants.items():
        if len(variants) > 1:
            print(f"Case variants for '{norm}': {variants}")

    # Find non-English tags (very basic heuristic)
    non_english = [tag for tag in all_tags if not tag.isascii()]
    print("Non-English tags:", non_english)


def normalize_tag(tag: str) -> str:
    """
    Normalize Unicode and lowercase
    NFKD stands for “Normalization Form Compatibility Decomposition.”
    It is one of the Unicode normalization forms, makes text strings represented in a consistent way
    """
    return unicodedata.normalize("NFKD", tag).lower().strip()


def parse_works_file(file_path: str = DEFAULT_FILENAME, line_limit: int = DEFAULT_LINE_LIMIT) -> list[Work]:
    """
    Parses the file into a list of Work dataclasses
    """
    ret = []

    with Timer(f"Parsing {file_path}"):
        with open(file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if line_limit and i >= line_limit:
                    break

                if debug():
                    if i % DEBUG_OUTPUT_QUANTA == 0:
                        print(f"loaded {i:,} items")

                work = parse_works_line(line)
                ret.append(work)

    return ret


def parse_works_line(line: str) -> Optional[Work]:
    """
    Parses a single line of the file into a single Work dataclass
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) < 5:
        return None

    work_type = parts[0]
    work_id = parts[1]
    revision = int(parts[2])
    last_modified = parts[3]

    try:
        # Decode the JSON
        parsed = orjson.loads(parts[4])

        # Map JSON fields to dataclass
        ret = Work(
            type=work_type,
            id=work_id,
            last_modified=last_modified,
            subjects=parsed.get("subjects", []),
            subject_places=parsed.get("subject_places", []),
            subject_times=parsed.get("subject_times", []),
            subject_people=parsed.get("subject_people", []),

            # revision=revision,
            # title=parsed.get("title"),
            # subtitle=parsed.get("subtitle"),
            # authors=parsed.get("authors", []),
            # translated_titles=parsed.get("translated_titles", []),
            # description=parsed.get("description"),
            # dewey_number=parsed.get("dewey_number", []),
            # lc_classifications=parsed.get("lc_classifications", []),
            # first_sentence=parsed.get("first_sentence"),
            # original_languages=parsed.get("original_languages", []),
            # other_titles=parsed.get("other_titles", []),
            # first_publish_date=parsed.get("first_publish_date"),
            # links=parsed.get("links", []),
            # notes=parsed.get("notes"),
            # cover_edition=parsed.get("cover_edition"),
            # covers=parsed.get("covers", []),
            # latest_revision=parsed.get("latest_revision"),
            # created=parsed.get("created"),
            # key=parsed.get("key"),
        )
        return ret

    except Exception:
        print(f"Error parsing line: {line}")
        return None


def download(
    url: str = DEFAULT_WORKS_URL, file_path: str = DEFAULT_FILENAME, line_limit: int = DEFAULT_LINE_LIMIT
) -> bool:
    """
    Downloads the OpenLibrary works dump and decompresses it. If error, returns False
    """
    try:
        with Timer("Downloading"):
            # Download the gzipped file
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download file: {response.status_code}")

            # Save the gzipped file locally
            gz_path = file_path + ".gz"
            with open(gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Decompress the gzipped file to the output path
        with Timer("Decompressing"):
            with gzip.open(gz_path, "rb") as f_in:
                with open(file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    except Exception as e:
        print(f"ERROR: {e}")
        return False

    return True


def set_debug(x: bool) -> None:
    global DEBUG
    DEBUG = x


def debug() -> bool:
    return DEBUG


def get_args(the_args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Returns the users's args
    """
    if not the_args:
        the_args = sys.argv[1:]

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--debug", action="store_true", default=DEBUG, help="Enable debug mode")
    arg_parser.add_argument("--download", action="store_true", default=False, help="Download a new file")
    arg_parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Filename to parse")
    arg_parser.add_argument("--limit", type=int, default=DEFAULT_LINE_LIMIT, help="Number of lines to parse")

    args, _ = arg_parser.parse_known_args(the_args)
    return args


if __name__ == "__main__":
    main()
