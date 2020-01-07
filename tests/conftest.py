import pytest

def pytest_addoption(parser):
    parser.addoption("--score_url", action="store",
        help="the score url of the ml web service")
    parser.addoption("--score_key", action="store",
        help="the score key of the ml web service")

@pytest.fixture
def score_url(request):
    return request.config.getoption("--score_url")

@pytest.fixture
def score_key(request):
    return request.config.getoption("--score_key")