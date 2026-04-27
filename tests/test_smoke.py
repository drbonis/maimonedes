from maimonedes import __version__


def test_package_imports_and_exposes_version() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""
