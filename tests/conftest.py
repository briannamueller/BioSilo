"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def root(tmp_path):
    """Give each test an isolated partition root."""
    return tmp_path
