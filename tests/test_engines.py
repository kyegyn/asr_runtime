from asr_runtime.engines.sensevoice import clean_text


def test_sensevoice_tags_removed_before_rich_postprocess() -> None:
    assert clean_text("<|zh|><|Speech|>打开灯。<|HAPPY|>") == "打开灯。"

