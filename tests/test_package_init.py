import importlib


def test_agentnet_package_exposes_version() -> None:
    agentnet = importlib.import_module("agentnet")

    assert agentnet.__version__ == "0.1.0"


def test_agentnet_version_module_defines_package_version() -> None:
    version_module = importlib.import_module("agentnet._version")

    assert version_module.__version__ == "0.1.0"
