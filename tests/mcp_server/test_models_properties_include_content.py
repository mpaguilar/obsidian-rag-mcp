"""Tests for DocumentResponse properties field and include_content parameter."""

import uuid
from datetime import datetime

from obsidian_rag.mcp_server.models import DocumentResponse, create_document_response


class _MockVault:
    def __init__(self, name: str = "Test Vault") -> None:
        self.name = name


class _MockDocument:
    def __init__(
        self,
        frontmatter_json: dict[str, object] | None = None,
        content: str = "# Content",
    ) -> None:
        self.id = uuid.uuid4()
        self.file_path = "path/to/doc.md"
        self.file_name = "doc.md"
        self.content = content
        self.frontmatter_json = frontmatter_json
        self.tags = ["tag1", "tag2"]
        self.created_at_fs = datetime.now()
        self.modified_at_fs = datetime.now()
        self.vault = _MockVault()


def test_document_response_has_properties_field() -> None:
    """DocumentResponse should accept and expose a properties field."""
    now = datetime.now()
    response = DocumentResponse(
        id=uuid.uuid4(),
        vault_name="Vault",
        file_path="path/doc.md",
        relative_path="path/doc.md",
        file_name="doc.md",
        content="content",
        kind=None,
        tags=[],
        similarity_score=0.0,
        created_at_fs=now,
        modified_at_fs=now,
        obsidian_uri="obsidian://open?vault=Vault&file=path%2Fdoc.md",
        properties={"author": "Alice"},
    )

    assert response.properties == {"author": "Alice"}


def test_document_response_properties_with_dict() -> None:
    """DocumentResponse properties should store arbitrary key-value pairs."""
    now = datetime.now()
    response = DocumentResponse(
        id=uuid.uuid4(),
        vault_name="Vault",
        file_path="path/doc.md",
        relative_path="path/doc.md",
        file_name="doc.md",
        content="content",
        kind="article",
        tags=["tag"],
        similarity_score=0.5,
        created_at_fs=now,
        modified_at_fs=now,
        obsidian_uri="obsidian://open?vault=Vault&file=path%2Fdoc.md",
        properties={"author": "Bob", "year": 2026, "nested": {"x": 1}},
    )

    assert response.properties["author"] == "Bob"
    assert response.properties["year"] == 2026
    assert response.properties["nested"] == {"x": 1}


def test_create_document_response_populates_properties_from_frontmatter_json() -> None:
    """create_document_response should populate properties from frontmatter_json."""
    doc = _MockDocument(frontmatter_json={"author": "Alice", "kind": "article"})

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.properties == {"author": "Alice", "kind": "article"}


def test_create_document_response_properties_none_when_no_frontmatter() -> None:
    """create_document_response should return None properties when frontmatter_json is None."""
    doc = _MockDocument(frontmatter_json=None)

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.properties is None


def test_create_document_response_properties_excludes_tags_key() -> None:
    """create_document_response should exclude the tags key from properties."""
    doc = _MockDocument(
        frontmatter_json={
            "tags": ["ignored"],
            "title": "My Title",
            "kind": "note",
        },
    )

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.properties == {"title": "My Title", "kind": "note"}


def test_create_document_response_include_content_true_preserves_content() -> None:
    """create_document_response with include_content=True should preserve document content."""
    doc = _MockDocument(content="original content")

    response = create_document_response(doc, 0.25, include_content=True)  # type: ignore[arg-type]

    assert response.content == "original content"


def test_create_document_content_false_produces_empty_string() -> None:
    """create_document_response with include_content=False should return empty content."""
    doc = _MockDocument(content="sensitive content")

    response = create_document_response(doc, 0.25, include_content=False)  # type: ignore[arg-type]

    assert response.content == ""


def test_create_document_include_content_false_does_not_affect_matching_chunk() -> None:
    """include_content=False should not affect the matching_chunk field."""
    doc = _MockDocument(content="content")

    response = create_document_response(  # type: ignore[arg-type]
        doc,
        0.25,
        matching_chunk="best chunk",
        include_content=False,
    )

    assert response.content == ""
    assert response.matching_chunk == "best chunk"


def test_create_document_include_content_default_is_true() -> None:
    """create_document_response should default to preserving content when include_content is omitted."""
    doc = _MockDocument(content="preserved")

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.content == "preserved"


def test_create_document_response_properties_empty_dict_when_frontmatter_empty() -> (
    None
):
    """create_document_response should return empty properties for empty frontmatter_json."""
    doc = _MockDocument(frontmatter_json={})

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.properties == {}


def test_properties_backward_compatibility_old_callers() -> None:
    """Old callers without include_content or properties should still work."""
    doc = _MockDocument(frontmatter_json={"title": "T"})

    response = create_document_response(doc, 0.25, "chunk")  # type: ignore[arg-type]

    assert response.content == doc.content
    assert response.matching_chunk == "chunk"
    assert response.properties == {"title": "T"}


def test_properties_backward_compatibility_none_default() -> None:
    """DocumentResponse should default properties to None when not provided."""
    now = datetime.now()
    response = DocumentResponse(
        id=uuid.uuid4(),
        vault_name="Vault",
        file_path="path/doc.md",
        relative_path="path/doc.md",
        file_name="doc.md",
        content="content",
        kind=None,
        tags=[],
        similarity_score=0.0,
        created_at_fs=now,
        modified_at_fs=now,
        obsidian_uri="obsidian://open?vault=Vault&file=path%2Fdoc.md",
    )

    assert response.properties is None
