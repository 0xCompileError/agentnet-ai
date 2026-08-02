import pytest

import agentnet as an


def test_markdown_representation_accepts_markdown_text() -> None:
    representation = an.MarkdownRepresentation()

    assert representation.identifier == "markdown"
    assert representation.media_type == "text/markdown"
    assert representation.validate("# Summary\n\n- Fast") == "# Summary\n\n- Fast"


def test_markdown_representation_can_require_heading() -> None:
    representation = an.MarkdownRepresentation(require_heading=True)

    assert representation.validate("# Summary\nText") == "# Summary\nText"

    with pytest.raises(an.AgentNetValidationError, match="heading"):
        representation.validate("Summary\nText", label="payload")


def test_markdown_representation_rejects_non_text_values() -> None:
    representation = an.MarkdownRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate({"summary": "ok"}, label="payload")


def test_markdown_representation_is_exported_from_package_root() -> None:
    assert an.MarkdownRepresentation is not None
