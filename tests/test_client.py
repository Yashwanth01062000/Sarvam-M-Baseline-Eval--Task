from src.sarvam_client import SarvamClient

def test_extract_content():
    data = {"choices": [{"message": {"content": "hello"}}]}
    assert SarvamClient._extract_content(data) == "hello"
