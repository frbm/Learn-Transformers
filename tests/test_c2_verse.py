from lm.verse.corpus import split_anthology

BODY = """

THE POWER OF LOVE
From "Valentinian"

Hear, ye ladies that despise
What the mighty Love has done;

Fear examples and be wise:
Fair Callisto was a nun;

John Fletcher [1579-1625]


ADVICE TO A LOVER


The sea hath many thousand sands,
The sun hath motes as many;


A preface paragraph without a title.
"""


def test_split_anthology_titles_authors_and_stanzas():
    p = split_anthology(BODY, "Various")
    assert [x["title"] for x in p] == ["THE POWER OF LOVE", "ADVICE TO A LOVER", "Untitled"]
    assert p[0]["author"] == "John Fletcher" and p[1]["author"] == "Various"
    assert p[0]["text"].count("\n\n") == 1                     # stanza breaks survive, the attribution does not
    assert "Valentinian" not in p[0]["text"] and "1579" not in p[0]["text"]
