from lib_ml import preprocess_text

def test_preprocess_text():
    raw = "I loved the FOOD!!! 😍😍"
    processed = preprocess_text(raw)
    assert isinstance(processed, str)
    assert "love" in processed  # assuming stemming reduces "loved" to "love"
    assert "food" in processed
    assert "😍" not in processed
