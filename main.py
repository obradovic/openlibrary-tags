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
from collections import defaultdict  # , Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import gzip
import shutil
import sys
import time
from typing import Iterable, Iterator, TypeVar
import unicodedata

import ciso8601
from lingua import Language, LanguageDetector, LanguageDetectorBuilder

# import matplotlib.pyplot as plt
import orjson
import requests


#
# GLOBALS
#
DEBUG = False
DEBUG_OUTPUT_QUANTA = 250_000
DEFAULT_LINE_LIMIT = 0
DEFAULT_FILENAME = "works.txt"
DEFAULT_WORKS_URL = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
LANGUAGE_DETECTOR: LanguageDetector | None = None


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
    # type: str     # This is always "/type/work"
    id: str
    last_modified: datetime

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
    # original_languages: Strings = field(default_factory=list)
    # other_titles: list[str] = field(default_factory=list)
    # first_publish_date: Optional[datetime] = None
    # links: list[dict] = field(default_factory=list)
    # notes: Optional[str] = None
    # cover_edition: Optional[dict] = None
    # covers: list[int] = field(default_factory=list)
    # latest_revision: Optional[int] = None
    created: datetime | None = None
    # key: Optional[str] = None


#
# TYPE ALIASES
#
T = TypeVar("T")
Works = list[Work]
Strings = list[str]
TagsToWorks = dict[str, Works]


#
# FUNCTIONS
#
def main():
    """
    This is main
    """
    args = get_args()
    set_debug(args.debug)
    initialize_language_detector()

    if args.download:
        ok = download(file_path=args.filename, line_limit=args.limit)
        if not ok:
            return

    works = parse_works_file(file_path=args.filename, line_limit=args.limit)
    analyze_tags(works)


def analyze_tags(works: Works):
    """
    Cursory analysis of tags
    """
    # Sometimes we have tags that are serialized dicts, fix those
    fix_dicts_in_subjects(works)

    # invert into a map of tags-to-works
    tags_to_works = index_works_by_subject(works)

    # sort by "number of items in the list"
    tags_by_size = sorted(tags_to_works.keys(), key=lambda tag: len(tags_to_works[tag]), reverse=True)
    tags_to_size = {x: len(tags_to_works[x]) for x in tags_by_size}

    # sort by keys in alphabetical order
    tags_by_alphabetical = sorted(tags_by_size)

    # get a list of the tags
    tags_general = [x.subjects for x in works if x.subjects]
    tags_people = [x.subject_people for x in works if x.subject_people]
    tags_places = [x.subject_places for x in works if x.subject_places]
    tags_times = [x.subject_times for x in works if x.subject_times]

    # flatten the list-of-lists-of-strings into lists-of-strings
    flattened_general = [x for y in tags_general for x in y if x]
    flattened_people = [x for y in tags_people for x in y if x]
    flattened_places = [x for y in tags_places for x in y if x]
    flattened_times = [x for y in tags_times for x in y if x]
    flattened_all = flattened_general + flattened_people + flattened_places + flattened_times

    # get counts
    count_general = len(flattened_general)
    count_people = len(flattened_people)
    count_places = len(flattened_places)
    count_times = len(flattened_times)
    count_all = len(flattened_all)
    print(f"General: {count_general:<12,} / {count_all:<14,} is {count_general / count_all:.1%}")
    print(f"People:  {count_people:<12,} / {count_all:<14,} is {count_people / count_all:.1%}")
    print(f"Places:  {count_places:<12,} / {count_all:<14,} is {count_places / count_all:.1%}")
    print(f"Times:   {count_times:<12,} / {count_all:<14,} is {count_times / count_all:.1%}")

    display_histogram(tags_to_size)

    # inspect the languages
    tags_to_languages, languages_to_tags = analyze_languages(tags_by_alphabetical)

    """
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
    """


def display_histogram(tags_to_size: dict[str, int], width: int = 50) -> None:
    """
    Shows an ASCII histogram of tag sizes with custom bins:
    0, 1, 2, 3, 4, 5, 6-10, 11-20, 21-100, 101+
    """
    if not tags_to_size:
        print("No data to display")
        return

    # Custom bins: (start, end) inclusive
    bins = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
        (11, 11),
        (12, 12),
        (13, 13),
        (14, 14),
        (15, 15),
        (16, 16),
        (17, 17),
        (18, 18),
        (19, 19),
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 69),
        (70, 79),
        (80, 89),
        (90, 99),
        (101, float("inf")),
    ]

    # Count how many tags fall into each bin
    bin_counts = [0] * len(bins)
    for size in tags_to_size.values():
        for i, (start, end) in enumerate(bins):
            if start <= size <= end:
                bin_counts[i] += 1
                break

    max_count = max(bin_counts)

    print("Histogram of how many works have this number of tags")
    for (start, end), count in zip(bins, bin_counts):
        if end == float("inf"):
            label = f"{start}+"
        else:
            label = f"{start}-{end}" if start != end else f"{start}"
        bar_length = int(count / max_count * width)
        bar = "#" * bar_length
        print(f"{label:>8} | {bar} ({count:,})")


def analyze_languages(tags: Strings) -> tuple[dict[str, Language], dict[Language, Strings]]:
    """
    Returns two dicts,
    """
    tags_to_languages = {}
    languages_to_tags = defaultdict(list)

    for tag in enumerate_progress(tags, "tags"):
        if tag:
            language = analyze_language(tag)

            tags_to_languages[tag] = language
            languages_to_tags[language].append(tag)

    return tags_to_languages, languages_to_tags


def analyze_language(string: str) -> Language | None:
    """
    Figures out what language this is in
    """
    if not LANGUAGE_DETECTOR:
        print("ERROR: Please initialize language detector")
        return None

    language = LANGUAGE_DETECTOR.detect_language_of(string)
    return language


def initialize_language_detector() -> None:
    """
    Initializes the language detector
    """
    global LANGUAGE_DETECTOR

    with Timer("Building language detector"):
        # Uses ALL languages
        LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

        # To specify the languages
        # LANGUAGES = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.ITALIAN]
        # LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()


def fix_dicts_in_subjects(works: Works) -> None:
    """
    If any of the tags are a dict not a string, fix that!
    """
    with Timer("Fixing dicts in tags"):
        for work in enumerate_progress(works, "works"):
            # for work in works:
            work.subjects = fix_dicts_in_list(work.subjects)
            work.subject_people = fix_dicts_in_list(work.subject_people)
            work.subject_places = fix_dicts_in_list(work.subject_places)
            work.subject_times = fix_dicts_in_list(work.subject_times)


def fix_dicts_in_list(strings: Strings) -> Strings:
    """
    If any of the strings is a dict, fix it
    """

    # if everything's a string (which happens most of the time) just return it
    if all(isinstance(x, str) for x in strings):
        return strings

    # otherwise make a new list
    ret: Strings = []
    for tag in strings:
        if isinstance(tag, dict):
            value = tag["value"]
            ret.append(value)
        else:
            ret.append(tag)

    return ret


def normalize_tag(tag: str) -> str:
    """
    Normalize Unicode and lowercase
    NFKD stands for “Normalization Form Compatibility Decomposition.”
    It is one of the Unicode normalization forms, makes text strings represented in a consistent way
    """
    return unicodedata.normalize("NFKD", tag).lower().strip()


def parse_works_file(file_path: str = DEFAULT_FILENAME, line_limit: int = DEFAULT_LINE_LIMIT) -> Works:
    """
    Parses the file into a list of Work dataclasses
    """
    ret = []

    with Timer(f"Parsing {file_path}"):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in enumerate_progress(file, f"lines from '{file_path}'"):
                work = parse_works_line(line)

                # Break if error!
                if not work:
                    print(f"ERROR: Could not parse line: {line}")
                    break

                ret.append(work)

    return ret


def parse_works_line(line: str) -> Work | None:
    """
    Parses a single line of the file into a single Work dataclass
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) < 5:
        return None

    # Many of these are commented-out to save memory!
    try:
        # work_type = parts[0]
        work_id = parts[1]
        # revision = int(parts[2])
        last_modified = ciso8601.parse_datetime(parts[3])

        # Decode the JSON
        parsed = orjson.loads(parts[4])

        # Map JSON fields to dataclass
        ret = Work(
            # type=work_type,
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
            # first_publish_date=parse_works_date(parsed, "first_publish_date"),
            # links=parsed.get("links", []),
            # notes=parsed.get("notes"),
            # cover_edition=parsed.get("cover_edition"),
            # covers=parsed.get("covers", []),
            # latest_revision=parsed.get("latest_revision"),
            created=parse_works_date(parsed, "created"),
            # key=parsed.get("key"),
        )
        return ret

    except Exception:
        print(f"Error parsing line: {line}")
        return None


def parse_works_date(parsed: dict, key: str) -> datetime | None:
    """
    Returns the parsed date, if it exists. None otherwise
    """
    type_value = parsed.get(key)
    if not type_value:
        return None

    value = type_value.get("value")
    if not value:
        return None

    if type_value.get("type") == "/type/datetime":
        return ciso8601.parse_datetime(value)

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


def index_works_by_subject(works: Works) -> TagsToWorks:
    """
    Returns a dict that maps tags to a list of each work that has the tag
    """
    ret = defaultdict(list)

    for work in works:
        tags = get_tags_for_work(work)
        for tag in tags:
            if not isinstance(tag, str):
                print(f"TAG TYPE: {type(tag)}: {tag}")
                break

            ret[tag].append(work)

    return ret


def get_tags_for_work(work: Work) -> Strings:
    """
    Returns all tags for a work, in a flattened list. This of course loses the information about which field a tag was in!
    """
    ret: Strings = []

    # tags can be in any of these fields
    subject_fields = [work.subjects, work.subject_people, work.subject_places, work.subject_times]

    for subject_field in subject_fields:
        for tag in subject_field:
            if tag:
                ret.append(tag)

    ret = sorted(ret)
    return ret


def enumerate_progress(iterable: Iterable[T], label: str = "items", quanta: int = DEBUG_OUTPUT_QUANTA) -> Iterator[T]:
    """
    Wraps an iterable and yields each element while printing progress every `quanta` iterations.
    """
    then = time.time()
    for i, item in enumerate(iterable, start=1):
        if i % quanta == 0:
            now = time.time()
            diff = now - then
            then = now
            print(f"{i:,} {label} took {diff:,.1f}s")

        yield item


def set_debug(x: bool) -> None:
    global DEBUG
    DEBUG = x


def debug() -> bool:
    return DEBUG


def get_args(the_args: Strings | None = None) -> argparse.Namespace:
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
