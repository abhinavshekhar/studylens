from core.ingestion import (
    chunk_pages,
    clean_text,
    is_scanned_or_empty_pdf,
    remove_repeated_headers_footers,
)


def test_clean_text_joins_hyphen_and_normalizes_whitespace() -> None:
    raw = "This is hyphen-\nated   text.\n\nWith\tmany   spaces."
    assert clean_text(raw) == "This is hyphenated text. With many spaces."


def test_remove_repeated_headers_footers_conservative() -> None:
    pages = [
        (1, "Header\nBody one\nFooter"),
        (2, "Header\nBody two\nFooter"),
        (3, "Header\nBody three\nFooter"),
    ]
    cleaned = remove_repeated_headers_footers(pages)
    assert cleaned == [(1, "Body one"), (2, "Body two"), (3, "Body three")]


def test_chunk_pages_preserves_overlap_and_page() -> None:
    words = [f"w{i}" for i in range(1, 11)]
    pages = [(2, " ".join(words))]
    chunks = chunk_pages("doc.txt", pages, size=5, overlap=2)

    assert [c.page for c in chunks] == [2, 2, 2, 2]
    assert chunks[0].text.split() == ["w1", "w2", "w3", "w4", "w5"]
    assert chunks[1].text.split()[:2] == ["w4", "w5"]
    assert all(c.text.strip() for c in chunks)


def test_chunk_pages_rejects_invalid_overlap() -> None:
    try:
        chunk_pages("doc", [(1, "a b c")], size=3, overlap=3)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_pages_skips_empty_page_text() -> None:
    chunks = chunk_pages("doc", [(1, "   \n\t"), (2, "value")], size=10, overlap=1)
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_scanned_pdf_detection() -> None:
    assert is_scanned_or_empty_pdf([(1, ""), (2, "\n\n")])
    assert not is_scanned_or_empty_pdf([(1, "hello")])
