"""
Examines OpenLibrary tags.
(Optionally) downloads the latest works file, and performs rudimentary analysis

SETUP:
  python -m textblob.download_corpora

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
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import gzip
import re
import shutil
import sys
import time
from typing import Iterable, Iterator, TypeVar
import unicodedata

# from autocorrect import Speller
import ciso8601
import inflect
from lingua import Language, LanguageDetector, LanguageDetectorBuilder
import orjson
import requests
from spellchecker import SpellChecker
from textblob import TextBlob


#
# TYPE ALIASES
#
Bin = tuple[float, float]
Bins = list[Bin]
T = TypeVar("T")
Strings = list[str]


#
# GLOBALS
#
DEBUG = False
DEBUG_OUTPUT_QUANTA = 250_000
DEFAULT_LINE_LIMIT = 0
DEFAULT_FILENAME = "works.txt"
DEFAULT_WORKS_URL = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
EMPTY = ""
ENGLISH_DETECTOR: LanguageDetector | None
LANGUAGE_DETECTOR: LanguageDetector | None

EMPTY = ""
LPAREN = "("
RPAREN = ")"

ANSI_ESCAPE_PATTERN = r"(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])"
ANSI_ESCAPE_REGEXP = re.compile(ANSI_ESCAPE_PATTERN)

BINS_TAGS_TO_COUNTS: Bins = [
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
    (100, 199),
    (200, 299),
    (300, 399),
    (400, 499),
    (500, float("inf")),
]

BINS_TAG_COUNTS_TO_WORKS: Bins = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 100),
    (101, float("inf")),
]


#
# TYPES
#

# @dataclass
# class AuthorRole:
#    type: dict
#    author: dict


@dataclass
class Work:
    # type: str     # This is always "/type/work"
    id: str
    last_modified: datetime

    subjects: Strings = field(default_factory=list)
    subject_places: Strings = field(default_factory=list)
    subject_times: Strings = field(default_factory=list)
    subject_people: Strings = field(default_factory=list)

    # revision: int
    # title: Optional[str] = None
    # subtitle: Optional[str] = None
    # authors: list[AuthorRole] = field(default_factory=list)
    # translated_titles: Strings = field(default_factory=list)
    # description: Optional[str] = None
    # dewey_number: Strings = field(default_factory=list)
    # lc_classifications: Strings = field(default_factory=list)
    # first_sentence: Optional[str] = None
    # original_languages: Strings = field(default_factory=list)
    # other_titles: Strings = field(default_factory=list)
    # first_publish_date: Optional[datetime] = None
    # links: list[dict] = field(default_factory=list)
    # notes: Optional[str] = None
    # cover_edition: Optional[dict] = None
    # covers: list[int] = field(default_factory=list)
    # latest_revision: Optional[int] = None
    created: datetime | None = None
    # key: Optional[str] = None

    # makes it hashable
    def __hash__(self) -> int:
        return hash(self.id)


Works = list[Work]
TagsToWorks = dict[str, Works]


@contextmanager
def Timer(name: str):
    start_time = time.time()

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


#
# FUNCTIONS
#
def main():
    """
    This is main
    """
    args = get_args()
    initialize(args)

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
    # Sanity check: sometimes we have tags that are serialized dicts, fix those
    fix_dicts_in_subjects(works)

    # invert into a map of tags-to-works
    tags_to_works = get_tags_to_works(works)

    # keep a list of all the tag strings, in alphabetical order
    tags = sorted(tags_to_works.keys())
    # tags_lowercase = sorted(list(set([x.lower() for x in tags])))

    # ###########################################
    #
    # QUESTION: WHAT TAGS START OR END WITH A SPACE?
    #
    # ###########################################
    print()
    with Timer("Analyzing whitespace"):
        tags_with_leading_spaces = [x for x in tags if x and x[0].isspace()]
        tags_with_trailing_spaces = [x for x in tags if x and x[-1].isspace()]
    print(f"Discovered {len(tags_with_leading_spaces):,} tags with leading spaces")
    print(f"Discovered {len(tags_with_trailing_spaces):,} tags with trailing spaces")

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS DOES EACH WORK HAVE?
    #
    # ###########################################
    print()
    with Timer("Analyzing tag counts per work"):
        # works_to_tag_count = {x: len(get_tags_for_work(x)) for x in works}
        works_to_tag_count = {x: get_tag_count_for_work(x) for x in works}
        tag_counts_to_works = dict(Counter(works_to_tag_count.values()))
        tag_counts_to_works = {x: tag_counts_to_works[x] for x in sorted(tag_counts_to_works.keys())}
    display_tag_count_histogram(tag_counts_to_works, title="Number of works with this many tags:")

    # ###########################################
    #
    # QUESTION: WHAT ARE THE MOST POPULAR TAGS?
    #
    # ###########################################
    print()
    with Timer("Analyzing tag counts per tag"):
        tags_by_tag_count = sorted(tags_to_works.keys(), key=lambda tag: len(tags_to_works[tag]), reverse=True)
        tags_to_tag_count = {x: len(tags_to_works[x]) for x in tags_by_tag_count}
    display_histogram(tags_to_tag_count.values(), BINS_TAGS_TO_COUNTS, title="Number of tags repeated this many times:")

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS OF WHAT TYPE ARE THERE
    #
    # ###########################################
    print()
    with Timer("Analyzing tag categories"):
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

    # ###########################################
    #
    # QUESTION: WHAT LANGUAGES ARE TAGS WRITTEN IN?
    #
    # ###########################################

    # NOTE: It detects English when they're not really? See "Čhulālongkō̜nmahāwitthayālai"
    print()
    with Timer("Analyzing tag languages"):
        tags_to_languages, languages_to_tags = analyze_languages(tags)
        languages_to_counts = {x: len(y) for x, y in languages_to_tags.items()}
        languages_to_counts_sorted = dict(sorted(languages_to_counts.items(), key=lambda x: x[1], reverse=True))
    for i, (language, count) in enumerate(languages_to_counts_sorted.items()):
        language_name = language.name.capitalize() if language else "Unknown"
        print(f"  {language_name:<14} has {count:,} tags")
        if i == 20:
            break

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS DIFFER ONLY BY CAPITALIZATION
    #
    # ###########################################
    print()
    with Timer("Analyzing capitalization"):
        capitalization_variants = get_capitalization_variants(tags)

        # Sort with "most variants" at the top
        capitalization_variants = dict(
            sorted(capitalization_variants.items(), key=lambda item: len(item[1]), reverse=True)
        )

        # All the dupes in a list
        dupe_tags = [x for y in capitalization_variants.values() for x in y if x]

        # How many works are affected?
        works_affected = []
        for tag in dupe_tags:
            works = tags_to_works[tag]
            works_affected.extend(works)
        works_affected_unique = list(set(works_affected))

    for i, (k, v) in enumerate(capitalization_variants.items()):
        if i == 10:
            break
        for j, variant in enumerate(sorted(v)):
            print(f"  Capitalization infraction {i}, variant {j}: '{variant}'")
        print()

    print(f"  {len(capitalization_variants):,} tags have capitalization variations")
    print(f"  {len(dupe_tags):,} unique tags that differ ONLY by capitalization")
    print(f"  {len(works_affected_unique):,} works would be affected by sanitizing capitalization")

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS HAVE DOUBLE-SPACES, or DOUBLE-PUNCTUATION, OR NEWLINES
    #
    # ###########################################
    with Timer("Analyzing punctuation errors"):
        double_dashes = [x for x in tags if "--" in x]
        double_spaces = [x for x in tags if "  " in x]
        newlines = [x for x in tags if "\n" in x]
        # at_sign = [x for x in tags if "@" in x]

        double_punctuation_pattern = (
            r"([!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])\1|([!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])\2"
        )
        double_punctuations = []
        for tag in tags:
            matches = re.findall(double_punctuation_pattern, tag)
            if matches:
                double_punctuations.append(tag)

    print(f"  {len(double_dashes):,} tags have double-dashes")
    print(f"  {len(double_spaces):,} tags have double-spaces")
    print(f"  {len(newlines):,} tags have newlines")
    print(f"  {len(double_punctuations):,} tags have double-punctuation characters")

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS DIFFER ONLY BY PLURALIZATION
    #
    # ###########################################
    # NOTE: the unimorph_inflect package could possibly help with non-English languages?
    with Timer("Analyzing pluralization errors"):
        print()
        english_tags = languages_to_tags[Language.ENGLISH]
        english_tags = get_strictly_english(english_tags)
        with Timer("Analyzing pluralization (English only)"):
            singular_to_plurals = get_plural_map(english_tags)

        print(f"  {len(singular_to_plurals):,} issues with pluralization:")
        for i, (singular, plurals) in enumerate(singular_to_plurals.items()):
            plurals_display = ", ".join(plurals)
            print(f"  {singular} has {len(plurals)} plurals: {plurals_display}")
            if i == 20:
                break

    # ###########################################
    #
    # QUESTION: HOW MANY STRINGS HAVE ANSI ESCAPE CODES?
    #
    # ###########################################
    with Timer("Analyzing ANSI escape codes"):
        escaped = {x: remove_ansi_escape_codes(x) for x in english_tags}
        escaped = {x: y for x, y in escaped.items() if x != y}
        escaped_frequencies = {x: tags_to_tag_count[x] for x in escaped}
        escaped_frequencies = dict(sorted(escaped_frequencies.items(), key=lambda x: x[1], reverse=True))
        escaped_count = sum(escaped_frequencies.values())

    print(f"  {len(escaped):,} unique tags have ANSI escape codes")
    print(f"  {escaped_count:,} total tags have ANSI escape codes")
    for i, (x, y) in enumerate(escaped_frequencies.items()):
        print(f"  {x} == {y}")
        if i == 20:
            break

    # ###########################################
    #
    # QUESTION: HOW MANY TAGS ARE ENCLOSED IN PARENTHESIS?
    #
    # ###########################################
    with Timer("Analyzing parenthesis"):
        have_parens = [x for x in tags if LPAREN in x or RPAREN in x]
        wrapped_parens = [x for x in have_parens if x[0] == LPAREN and x[-1] == RPAREN]
    print(f"  {len(have_parens):,} tags have parenthesis")
    print(f"  {len(wrapped_parens):,} tags are wrapped in parenthesis")

    # ###########################################
    #
    # QUESTION: HOW MANY MISSPELLINGS ARE THERE?
    #
    # ###########################################
    with Timer("Analyzing misspellings"):
        tags_to_corrections = get_misspellings(english_tags)
    print(tags_to_corrections)


def get_misspellings(tags: Strings) -> dict[str, str]:
    """
    Returns a dict of words to a list of their misspellings
    """
    checker = SpellChecker()
    # autocorrecter = Speller()

    ret = {}

    for tag in tags:
        blob = TextBlob(tag)
        words = blob.words
        word_tags = blob.tags

        # ignore proper nouns.
        # Proper nouns are typically tagged as NNP (singular proper noun) or NNPS (plural proper noun).
        words = [x[0] for x in word_tags if x[1] not in ["NNP", "NNPS"]]

        # Get a list of the misspelled words in this tag
        for word in words:
            misspelled = checker.unknown(word)
            if misspelled:
                ret[misspelled] = word

        """
        corrected = autocorrecter(tag)
        if corrected == tag:
            continue
        """

    return ret


#
# HISTOGRAMS
#
def display_histogram(counts: Iterable[int], bins: Bins, width: int = 50, title: str = "") -> None:
    """
    Histogram of raw values (each value contributes weight 1)
    """
    counts = list(counts)
    if not counts:
        print("No data to display")
        return

    bin_counts = compute_bin_counts(counts, [1] * len(counts), bins)
    print_ascii_histogram(bins, bin_counts, width, title)


def display_tag_count_histogram(data: dict[int, int], bins: Bins = [], width: int = 50, title: str = "") -> None:
    """
    Histogram where each tag_count has an associated weight = num tags
    """
    if not data:
        print("No data to display")
        return

    bins = bins or BINS_TAG_COUNTS_TO_WORKS

    values = data.keys()
    weights = data.values()

    bin_counts = compute_bin_counts(values, weights, bins)
    print_ascii_histogram(bins, bin_counts, width, title)


def compute_bin_counts(values: Iterable[int], weights: Iterable[int], bins: Bins) -> list[int]:
    """
    Assign values (with weights) to bins and return weighted bin counts
    """
    bin_counts = [0] * len(bins)
    for val, weight in zip(values, weights):
        for i, (start, end) in enumerate(bins):
            if start <= val <= end:
                bin_counts[i] += weight
                break
    return bin_counts


def print_ascii_histogram(bins: Bins, bin_counts: list[int], width: int, title: str) -> None:
    """
    Print the ASCII histogram
    """
    max_count = max(bin_counts) if bin_counts else 0

    title = title or "Histogram:"
    print(title)

    for (start, end), count in zip(bins, bin_counts):
        if end == float("inf"):
            label = f"{start}+"
        else:
            label = f"{start}" if start == end else f"{start}-{end}"

        bar_length = int(count / max_count * width) if max_count else 0
        bar = "#" * bar_length
        print(f"{label:>8} | {bar} ({count:,})")


#
# LANGUAGES
#
def initialize_language_detector() -> None:
    """
    Initializes the language detector
    """
    global ENGLISH_DETECTOR, LANGUAGE_DETECTOR

    with Timer("Building language detector"):
        # Uses ALL languages
        ENGLISH_DETECTOR = LanguageDetectorBuilder.from_languages(Language.ENGLISH).build()
        LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
        # LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


def analyze_languages(tags: Strings) -> tuple[dict[str, Language], dict[Language, Strings]]:
    """
    Returns two dicts, the second being the inverse of the first.
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
    if ENGLISH_DETECTOR is None or LANGUAGE_DETECTOR is None:
        print("ERROR: Please initialize language detector")
        return None

    """
    # first we try english, most of the tags are english and lingua's a little wierd about this?
    language = ENGLISH_DETECTOR.detect_language_of(string)
    if language == Language.ENGLISH:
        return language
    """

    # if its NOT english, try to infer what it is
    language = LANGUAGE_DETECTOR.detect_language_of(string)
    return language


#
# PARSING AND DOWNLOADING WORKS FILES
#
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


#
# MISCELLANEOUS HELPERS
#


def contains_ansi_escape_codes(string: str) -> bool:
    """
    Returns True if the string contains an ANSI escape code. False otherwise
    """
    new = remove_ansi_escape_codes(string)
    return new != string


def remove_ansi_escape_codes(string: str) -> str:
    """
    Removes ANSI escape codes, if they exist
    """
    return ANSI_ESCAPE_REGEXP.sub(EMPTY, string)


def get_strictly_english(strings: Strings) -> Strings:
    """
    Return the tags that are strictly ASCII english
    """
    return [x for x in strings if x.isascii()]


def get_plural_map(tags: Strings) -> dict[str, Strings]:
    """
    Figures out what the plurals are
    """
    tags = sorted(list(set([x.lower() for x in tags])))  # Lowercase everything
    tags = [x.strip() for x in tags]  # Remove leading/trailing whitespace

    tags_set = set(tags)  # For O(1) testing
    inflector = inflect.engine()

    ret = defaultdict(list)
    for tag in tags:
        if " " in tag or "--" in tag:
            continue

        singular = inflector.singular_noun(tag)

        # if this is already singular
        if singular == tag:
            continue

        if singular in tags_set:
            if tag not in ret[singular]:
                ret[singular].append(tag)

    # Remove anything that doesnt have plural variants
    ret = {x: y for x, y in ret.items() if len(y) > 1}

    return ret


def get_capitalization_variants(strings: Strings) -> dict[str, Strings]:
    """
    Returns a dict mapping the lowercase string to a list of all its capitalization variants
    """
    ret = defaultdict(list)

    for string in strings:
        key = string.lower()

        if string not in ret[key]:
            ret[key].append(string)

    # Remove items that only have one variant (no capitalization issues)
    ret = {x: y for x, y in ret.items() if len(y) > 1}

    return ret


def get_tags_to_works(works: Works) -> TagsToWorks:
    """
    Returns a dict that maps tags to a list of each work that has the tag
    """
    ret = defaultdict(list)

    for work in works:
        tags = get_tags_for_work(work)
        """
        if not tags:
            ret[EMPTY].append(work)
        """

        for tag in tags:
            ret[tag].append(work)

    return ret


def get_tags_for_work(work: Work) -> Strings:
    """
    Returns all tags for a work, in a flattened list. This of course loses the information about which field a tag was in!
    """
    ret = []

    ret.extend(work.subjects)
    ret.extend(work.subject_people)
    ret.extend(work.subject_places)
    ret.extend(work.subject_times)

    return ret


def get_tag_count_for_work(work: Work) -> int:
    """
    Returns the number of tags this work has
    """
    return len(work.subjects) + len(work.subject_people) + len(work.subject_places) + len(work.subject_times)


def get_tag_counts_to_works(works: Works) -> dict[int, Works]:
    """
    Returns a dict of tag counts to works that have that tag count
    """
    ret = defaultdict(list)

    for work in works:
        # tags = get_tags_for_work(work)
        # tag_count = len(tags)
        tag_count = get_tag_count_for_work(work)

        ret[tag_count].append(work)

    return ret


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
    If any of the strings is a dict, fix it. There's like a couple-dozen instances of this, total
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


def initialize(args: argparse.Namespace) -> None:
    set_debug(args.debug)
    initialize_language_detector()


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
