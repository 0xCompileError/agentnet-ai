import pytest

import agentnet as an


def test_xml_representation_accepts_well_formed_xml() -> None:
    representation = an.XMLRepresentation()

    xml = "<summary><item>ok</item></summary>"

    assert representation.identifier == "xml"
    assert representation.media_type == "application/xml"
    assert representation.validate(xml) == xml


def test_xml_representation_can_require_root_tag() -> None:
    representation = an.XMLRepresentation(root_tag="summary")

    assert representation.validate("<summary>ok</summary>") == "<summary>ok</summary>"

    with pytest.raises(an.AgentNetValidationError, match="root tag"):
        representation.validate("<report>ok</report>", label="payload")


def test_xml_representation_rejects_invalid_xml() -> None:
    representation = an.XMLRepresentation()

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate("<summary>", label="payload")

    with pytest.raises(an.AgentNetValidationError, match="payload"):
        representation.validate({"summary": "ok"}, label="payload")


def test_xml_representation_is_exported_from_package_root() -> None:
    assert an.XMLRepresentation is not None
