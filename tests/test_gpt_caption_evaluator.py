from unittest import mock

import gpt_caption_evaluator as gce


def test_evaluate_caption_returns_score():
    mocked_response = {"choices": [{"message": {"content": "7"}}]}
    with mock.patch("openai.ChatCompletion.create", return_value=mocked_response) as create_mock:
        score = gce.evaluate_caption("A caption", api_key="key")
        assert score == 7.0
        create_mock.assert_called_once()


def test_evaluate_file_scores_multiple_captions(tmp_path):
    path = tmp_path / "caps.txt"
    path.write_text("cap1\ncap2\n", encoding="utf-8")

    responses = [
        {"choices": [{"message": {"content": "3"}}]},
        {"choices": [{"message": {"content": "9"}}]},
    ]

    with mock.patch("openai.ChatCompletion.create", side_effect=responses):
        results = gce.evaluate_file(str(path), api_key="key")

    assert results == [
        {"caption": "cap1", "score": 3.0},
        {"caption": "cap2", "score": 9.0},
    ]

