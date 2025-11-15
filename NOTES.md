NOTES:
    Use consistent naming:
        all lowercase
        all singular no plural
        all english (?? and translate in realtime as needed ??)

    Keep track of both "as-entered tags" and "sanitized tags"
        Prefer sanitized tags when searching
        Provide an explanation/audit trail of how a tag was sanitized, and what rule(s) were used and when

    Version sanitization rules (use Semantic Versioning, with a path from that to timestamp . git hash and what was changed)
        When sanitization rules change, re-run

    All english?
        "literatura" -> "literature'joodse literatuur':"
        "joodse literatuur" -> "jewish" and "literature"    # this is dutch

    Avoid overlapping tags? (ie no "Fantasy/Sci-Fi", just use "Fantasy" (and use Solr synonyms?)

    Avoid compound tags, break them down into multiple tags
        "Islam -- Relations -- Christianity" -> "Islam" and "Relations" and "Christianity"
        (somehow remove "Relations"? doesnt seem to add much here)

    What to do with "roach (fish)"?
        Decompose into "roach" and "fish" ?

    Use dash, not underscore

    no punctuation characters - remove commas

    whitespace is ok (ie "series:harry potter")
        But avoid leading / trailing whitespace

    Use singular case, not plural (Solr can then adjust when searching)
        "single mothers" -> "single mother"

    Use colon as a separator character (general : specific | more-specific)
        But sometimes specificity is unclear!
            "chinese dramatists" could reasonably be either "dramatists:chinese" or "chinese:dramatists"
            In this case, split into two tags, "chinese" and "dramatists"

    If there will only ever be one book with this tag, it should not be a tag
        ie dont tag "isbn:234234234"

    Define synonyms in Solr:
        cp == "chinese poetry" ?
        mine parentheses for synonyms
        find clusters of tags

    Suggest tags, based on content or summary analysis, or wikipedia metadata analysis, or other books by the author(s)
    Which books or authors have similar tags?
    Should we tag authors as well?

    Create a slack bot to:
        /tag chinese
            will give you a book to tag that has an existing tag with the "chinese"

    Accept free-form input for tags, but then validate it and present it for approval to the librarian

    Keep track of who has sanitized tags
    Keep track of deleted / modified tags?  So we can keep a "tagging history", revert back to previous tags, track tag migration over time?


TAXONOMY:
    language:
    language-original:
    translated-from:

    award:              Hugo:Fiction:Fiction:1980 or New York Times
    series:             "harry potter" or "standalone"
    edition:
    genre:              find
