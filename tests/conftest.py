import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module")
def client():
    """
    provides a managed TestClient instance for API integration tests.
    """
    with TestClient(app) as test_client:
        yield test_client
