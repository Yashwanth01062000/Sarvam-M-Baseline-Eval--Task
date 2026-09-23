from src.scoring import exact_match, reasoning_score, summarization_score

def test_exact_match():
    assert exact_match(" New Delhi. ", "new delhi") == 1
    assert exact_match("Mumbai", "Delhi") == 0

def test_reasoning():
    assert reasoning_score("The answer is 42.", "42") == 1
    assert reasoning_score("The answer is 41.", "42") == 0

def test_summary():
    assert summarization_score(4, 5) == 4.5
