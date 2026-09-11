from mi_reward.inference.qwen3_vl_reward import DEFAULT_SCORE_MAP, _parse_label


def test_parse_label_exact_and_embedded() -> None:
    assert _parse_label("Positive") == "Positive"
    assert _parse_label("  Unclear\n") == "Unclear"
    assert _parse_label("The label is Negative.") == "Negative"


def test_parse_label_rejects_ambiguous_output() -> None:
    assert _parse_label("Positive or Negative") is None
    assert _parse_label("unknown") is None


def test_default_score_map_is_ordered() -> None:
    assert DEFAULT_SCORE_MAP["Positive"] > DEFAULT_SCORE_MAP["Unclear"]
    assert DEFAULT_SCORE_MAP["Unclear"] > DEFAULT_SCORE_MAP["Negative"]
