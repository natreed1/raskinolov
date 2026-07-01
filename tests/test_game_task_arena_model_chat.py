"""Unit tests for the Model Chat agentic-loop + judge-voting helpers.

These cover the pure/testable module-level functions in `game_task_arena.py`
(stop-signal detection, judge verdict parsing, vote tallying, transcript
name-keying, and debater/judge filtering). No model backend is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import game_task_arena as gta  # noqa: E402


def _speaker(name: str, *, is_judge: bool = False, backend: str = "Local") -> dict:
    return {"name": name, "backend": backend, "profile": "", "is_judge": is_judge}


def test_debate_signaled_stop_detects_token_case_insensitively():
    assert gta._debate_signaled_stop("I agree with you.\n[[debate_concluded]]")
    assert gta._debate_signaled_stop("Sounds good.\n[[DEBATE_CONCLUDED]]")
    assert not gta._debate_signaled_stop("I still disagree on the risk assessment.")


def test_strip_stop_token_removes_token_and_trims_whitespace():
    text = "We're aligned now.\n\n[[DEBATE_CONCLUDED]]\n"
    cleaned = gta._strip_stop_token(text)
    assert gta.DEBATE_STOP_TOKEN not in cleaned
    assert cleaned == "We're aligned now."


def test_strip_stop_token_noop_when_absent():
    text = "Nothing to strip here."
    assert gta._strip_stop_token(text) == text


def test_debater_and_judge_name_filters():
    room = [_speaker("Alice"), _speaker("Bob"), _speaker("Referee", is_judge=True)]
    assert gta._debater_names(room) == ["Alice", "Bob"]
    assert gta._judge_names(room) == ["Referee"]


def test_transcript_speaker_names_dedupes_and_excludes_human():
    chat_state = [
        {"speaker_name": "Human", "content": "Go!"},
        {"speaker_name": "Alice", "content": "Point one."},
        {"speaker_name": "Bob", "content": "Rebuttal."},
        {"speaker_name": "Alice", "content": "Point two."},
    ]
    assert gta._transcript_speaker_names(chat_state) == ["Alice", "Bob"]


def test_chat_transcript_by_name_uses_bare_names():
    chat_state = [
        {"speaker": "Alice [Local: some/adapter on some-model]", "speaker_name": "Alice", "content": "Hello"},
        {"speaker": "Bob [Frontier: gpt]", "speaker_name": "Bob", "content": "Hi back"},
    ]
    transcript = gta._chat_transcript_by_name(chat_state)
    assert "Alice: Hello" in transcript
    assert "Bob: Hi back" in transcript
    assert "[Local:" not in transcript


def test_parse_judge_verdict_matches_candidate_name():
    raw = "WINNER: Alice\nREASONING: Alice cited concrete evidence and addressed Bob's rebuttal directly."
    result = gta._parse_judge_verdict(raw, ["Alice", "Bob"])
    assert result["winner"] == "Alice"
    assert result["parse_ok"] is True
    assert "evidence" in result["reasoning"]


def test_parse_judge_verdict_tolerates_extra_text_on_winner_line():
    raw = "WINNER: Alice (the Local implementer)\nREASONING: Stronger verification story."
    result = gta._parse_judge_verdict(raw, ["Alice", "Bob"])
    assert result["winner"] == "Alice"


def test_parse_judge_verdict_handles_tie_language():
    raw = "WINNER: TIE\nREASONING: Both made equally strong points."
    result = gta._parse_judge_verdict(raw, ["Alice", "Bob"])
    assert result["winner"] == "tie"
    assert result["parse_ok"] is True


def test_parse_judge_verdict_unparseable_returns_parse_ok_false():
    raw = "I think this was a close debate."
    result = gta._parse_judge_verdict(raw, ["Alice", "Bob"])
    assert result["winner"] is None
    assert result["parse_ok"] is False


def test_tally_judge_votes_majority():
    verdicts = [
        {"judge": "J1", "winner": "Alice"},
        {"judge": "J2", "winner": "Alice"},
        {"judge": "J3", "winner": "Bob"},
    ]
    tally = gta._tally_judge_votes(verdicts)
    assert tally["winner"] == "Alice"
    assert tally["tie"] is False
    assert tally["counts"] == {"Alice": 2, "Bob": 1}
    assert tally["unparsed"] == 0


def test_tally_judge_votes_tie():
    verdicts = [
        {"judge": "J1", "winner": "Alice"},
        {"judge": "J2", "winner": "Bob"},
    ]
    tally = gta._tally_judge_votes(verdicts)
    assert tally["tie"] is True
    assert tally["winner"] is None
    assert set(tally["leaders"]) == {"Alice", "Bob"}


def test_tally_judge_votes_counts_unparsed_and_ignores_them():
    verdicts = [
        {"judge": "J1", "winner": "Alice"},
        {"judge": "J2", "winner": None},
    ]
    tally = gta._tally_judge_votes(verdicts)
    assert tally["winner"] == "Alice"
    assert tally["unparsed"] == 1


def test_tally_judge_votes_no_votes():
    tally = gta._tally_judge_votes([{"judge": "J1", "winner": None}])
    assert tally["winner"] is None
    assert tally["tie"] is False
    assert tally["counts"] == {}
    assert tally["unparsed"] == 1


def test_render_judge_scoreboard_reports_winner_and_votes():
    verdicts = [
        {"judge": "J1", "judge_label": "J1 [Local]", "winner": "Alice", "reasoning": "Better evidence."},
        {"judge": "J2", "judge_label": "J2 [Frontier]", "winner": "Bob", "reasoning": "More concise."},
        {"judge": "J3", "judge_label": "J3 [Local]", "winner": "Alice", "reasoning": "Addressed rebuttal."},
    ]
    tally = gta._tally_judge_votes(verdicts)
    board = gta._render_judge_scoreboard(verdicts, tally, ["Alice", "Bob"])
    assert "Alice wins" in board
    assert "Alice" in board and "Bob" in board


def test_build_debate_record_shape_and_ids():
    room_state = [_speaker("Alice"), _speaker("Bob"), _speaker("Referee", is_judge=True)]
    chat_state = [
        {"speaker": "Human", "speaker_name": "Human", "content": "Debate: is X better than Y?"},
        {"speaker": "Alice [Local]", "speaker_name": "Alice", "content": "X is better because..."},
        {"speaker": "Bob [Frontier]", "speaker_name": "Bob", "content": "Actually Y wins because..."},
    ]
    verdicts = [{"judge": "Referee", "judge_label": "Referee [Local]", "winner": "Alice", "reasoning": "..."}]
    tally = gta._tally_judge_votes(verdicts)
    record = gta._build_debate_record(
        room_state=room_state,
        chat_state=chat_state,
        judge_names=["Referee"],
        verdicts=verdicts,
        tally=tally,
    )
    assert record["debate_id"].startswith("model_chat_alice-vs-bob_")
    assert record["winner"] == "Alice"
    assert record["tie"] is False
    assert len(record["transcript"]) == 3
    assert record["judges"] == ["Referee"]
    assert {"name": "Referee", "backend": "Local", "is_judge": True} in record["speakers"]
