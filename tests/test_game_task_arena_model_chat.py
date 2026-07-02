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


def test_debate_signaled_pass_detects_token_case_insensitively():
    assert gta._debate_signaled_pass("[[PASS]]")
    assert gta._debate_signaled_pass("Nothing new.\n[[pass]]")
    assert not gta._debate_signaled_pass("I still have a new argument.")


def test_strip_pass_token_removes_token_and_trims_whitespace():
    text = "No new evidence.\n\n[[PASS]]\n"
    cleaned = gta._strip_pass_token(text)
    assert gta.DEBATE_PASS_TOKEN not in cleaned
    assert cleaned == "No new evidence."


def test_is_debate_pass_turn_detects_explicit_pass_and_aliases():
    assert gta._is_debate_pass_turn("", "[[PASS]]")
    assert gta._is_debate_pass_turn("pass")
    assert not gta._is_debate_pass_turn("War and Peace shows Tolstoy's panoramic realism.")


def test_parse_debate_model_output_omits_explicit_pass():
    speaker = {"name": "Researcher", "profile": "Role: researcher."}
    content, ended, explicit_pass = gta._parse_debate_model_output(
        speaker,
        "[[PASS]]",
        [],
        ["Researcher", "Risk Auditor"],
    )
    assert content is None
    assert explicit_pass
    assert not ended


def test_parse_debate_model_output_accepts_quality_turn():
    speaker = {"name": "Tolstoy Advocate", "profile": "Argue Tolstoy is greatest."}
    text = (
        "REBUTTAL: Scale matters. NEW EVIDENCE: War and Peace spans five families in 1812. "
        "CLAIM: Tolstoy is Russia's greatest novelist."
    )
    content, ended, explicit_pass = gta._parse_debate_model_output(speaker, text, [], ["Tolstoy Advocate"])
    assert content == text
    assert not explicit_pass


def test_generic_fallback_returns_opening_on_first_turn():
    speaker = {"name": "Researcher", "profile": "Role: researcher. Evidence-first."}
    fallback = gta._fallback_debate_turn(speaker, [])
    assert fallback != gta.DEBATE_PASS_TOKEN
    assert "CLAIM:" in fallback
    assert "Tolstoy" in fallback or "Dostoevsky" in fallback


def test_generic_fallback_passes_on_later_turn():
    speaker = {"name": "Researcher", "profile": "Role: researcher. Evidence-first."}
    prior = [
        {"speaker_name": "Human", "content": "Who is the greatest Russian author?"},
        {"speaker_name": "Researcher", "content": gta.GENERIC_DEBATE_OPENING_FALLBACKS[0]},
    ]
    assert gta._fallback_debate_turn(speaker, prior) == gta.DEBATE_PASS_TOKEN


def test_fallback_exhausted_advocate_options_returns_pass():
    speaker = {"name": "Tolstoy Advocate", "profile": ""}
    prior = [
        {"speaker_name": "Tolstoy Advocate", "content": gta.DEBATE_FALLBACK_TURNS["Tolstoy Advocate"][0]},
        {"speaker_name": "Tolstoy Advocate", "content": gta.DEBATE_FALLBACK_TURNS["Tolstoy Advocate"][1]},
        {"speaker_name": "Tolstoy Advocate", "content": gta.DEBATE_FALLBACK_TURNS["Tolstoy Advocate"][2]},
    ]
    assert gta._fallback_debate_turn(speaker, prior) == gta.DEBATE_PASS_TOKEN


def test_rule_based_conversation_summary_covers_topic_and_positions():
    chat_state = [
        {"speaker_name": "Human", "content": "Who is the greatest Russian author?"},
        {
            "speaker_name": "Tolstoy Advocate",
            "content": "CLAIM: Tolstoy is Russia's greatest novelist because of War and Peace.",
        },
        {
            "speaker_name": "Dostoevsky Advocate",
            "content": "CLAIM: Dostoevsky is greatest because Crime and Punishment maps guilt.",
        },
    ]
    summary = gta._rule_based_conversation_summary(
        chat_state,
        [],
        stop_reason="Reached max rounds (2).",
    )
    assert "centered around" in summary.lower()
    assert "greatest Russian author" in summary
    assert "Tolstoy Advocate" in summary
    assert "Dostoevsky Advocate" in summary
    assert "The basis of Tolstoy Advocate's argument was" in summary
    assert "The basis of Dostoevsky Advocate's argument was" in summary
    assert summary.strip().startswith("This conversation")
    assert "In summary" in summary
    assert "end condition" not in summary.lower()
    assert "reached max rounds" not in summary.lower()
    assert "\n\n" in summary


def test_rule_based_conversation_summary_includes_judge_winner_in_closing():
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {"speaker_name": "Alice", "content": "CLAIM: Tolstoy for panoramic realism."},
        {"speaker_name": "Bob", "content": "CLAIM: Dostoevsky for psychological depth."},
    ]
    summary = gta._rule_based_conversation_summary(
        chat_state,
        [],
        tally={"winner": "Alice"},
    )
    assert "In summary" in summary
    assert "Alice" in summary
    assert "stronger overall case" in summary


def test_build_conversation_summary_request_uses_narrative_structure():
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {"speaker_name": "Alice", "content": "CLAIM: Tolstoy for War and Peace."},
        {"speaker_name": "Bob", "content": "CLAIM: Dostoevsky for Karamazov."},
    ]
    req = gta._build_conversation_summary_request(
        chat_state,
        stop_reason="Completed 3 debate phase(s).",
        tally={"winner": "Alice"},
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "Opening paragraph" in system or "Opening paragraph" in user
    assert "The basis of" in user
    assert "In summary" in user
    assert "End condition: Completed" not in user
    assert "inline speaker labels" in user.lower()


def test_normalize_summary_text_preserves_paragraph_breaks():
    text = "First paragraph here.\n\nSecond paragraph here."
    normalized = gta._normalize_summary_text(text)
    assert normalized == "First paragraph here.\n\nSecond paragraph here."


def test_format_debate_summary_panel_prefixes_heading():
    panel = gta._format_debate_summary_panel("Short synopsis here.")
    assert panel.startswith("### Conversation summary")
    assert "Short synopsis here." in panel


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


def test_strip_leading_speaker_prefix_removes_known_names_only():
    names = ["Calm", "Aggressive"]
    assert gta._strip_leading_speaker_prefix("Aggressive: I disagree.", names) == "I disagree."
    # Repeated / cross-speaker prefixes are stripped.
    assert gta._strip_leading_speaker_prefix("Aggressive: Calm: Point.", names) == "Point."
    # Unknown lead-ins are preserved.
    assert gta._strip_leading_speaker_prefix("Note: this matters.", names) == "Note: this matters."


def test_truncate_at_next_speaker_keeps_only_current_turn():
    names = ["Calm", "Aggressive"]
    reply = (
        "Incremental refactoring is safer and testable.\n\n"
        "Human: But what about speed?\n\n"
        "Aggressive: A rewrite is faster."
    )
    truncated = gta._truncate_at_next_speaker(reply, names)
    assert truncated == "Incremental refactoring is safer and testable."


def test_truncate_at_next_speaker_noop_without_embedded_turns():
    names = ["Calm", "Aggressive"]
    reply = "A single coherent argument with no embedded turns."
    assert gta._truncate_at_next_speaker(reply, names) == reply


def test_strip_trailing_bracket_marker_removes_garbled_stop_tokens():
    assert gta._strip_trailing_bracket_marker("We agree. [[DEBATED]]") == "We agree."
    assert gta._strip_trailing_bracket_marker("Done here. [[DEBATE_CONCLUDED]") == "Done here."
    assert gta._strip_trailing_bracket_marker("No marker present.") == "No marker present."


def test_clean_debate_reply_end_to_end():
    names = ["Calm", "Aggressive"]
    raw = (
        "Aggressive: A full rewrite unlocks a cleaner architecture.\n\n"
        "Human: Isn't that risky?\n\n"
        "Calm: Yes, very risky. [[DEBATED]]"
    )
    cleaned = gta._clean_debate_reply(raw, names)
    assert cleaned == "A full rewrite unlocks a cleaner architecture."


def test_strip_bracket_speaker_hallucinations_keeps_first_chunk():
    names = ["Tolstoy Advocate", "Dostoevsky Advocate"]
    raw = (
        '[Dostoevsky Advocate: "Crime and Punishment" is greatest.] '
        "War and Peace maps society at epic scale."
        '\n[Dostoevsky Advocate: another fake turn.] More text.'
    )
    cleaned = gta._strip_bracket_speaker_hallucinations(raw, names)
    assert cleaned == "War and Peace maps society at epic scale."


def test_has_author_mixups_detects_wrong_pairings():
    assert gta._has_author_mixups("Tolstoy's Crime and Punishment is deep.")
    assert gta._has_author_mixups("Dostoevsky's Anna Karenina remains unmatched.")
    assert not gta._has_author_mixups("Tolstoy's War and Peace is an epic.")


def test_violates_stance_lock_detects_wrong_side():
    tolstoy = {"name": "Tolstoy Advocate", "profile": "Argue Leo Tolstoy is greatest."}
    dost = {"name": "Dostoevsky Advocate", "profile": "Argue Dostoevsky is greatest."}
    assert gta._violates_stance_lock(
        tolstoy,
        "Dostoevsky's Crime and Punishment positions him as the greatest Russian author.",
    )
    assert not gta._violates_stance_lock(
        tolstoy,
        "War and Peace makes Tolstoy the greatest Russian novelist.",
    )
    assert gta._violates_stance_lock(
        dost,
        "Tolstoy is the greater novelist because of Anna Karenina.",
    )


def test_turn_quality_ok_rejects_repetition_and_mixups():
    speaker = {"name": "Tolstoy Advocate", "profile": "Argue Leo Tolstoy is greatest."}
    prior = [{"speaker_name": "Dostoevsky Advocate", "content": "Crime and Punishment probes guilt."}]
    assert gta._turn_quality_ok(
        speaker,
        "War and Peace shows Tolstoy's panoramic realism and moral scope.",
        prior,
    )
    assert not gta._turn_quality_ok(
        speaker,
        "Dostoevsky's Anna Karenina is the greatest Russian novel.",
        prior,
    )
    assert not gta._turn_quality_ok(
        speaker,
        "Crime and Punishment probes guilt in St. Petersburg.",
        prior,
    )


def test_debate_search_snippets_uses_curated_offline(monkeypatch):
    monkeypatch.delenv("FE_DEBATE_SEARCH", raising=False)
    block = gta._debate_search_snippets("Leo Tolstoy War and Peace significance")
    assert "War and Peace" in block
    assert block.startswith("- ")


def test_debate_reference_facts_includes_speaker_stance(monkeypatch):
    monkeypatch.setenv("FE_DEBATE_SEARCH", "1")
    speaker = {"name": "Dostoevsky Advocate", "profile": "Argue Dostoevsky is greatest."}
    facts = gta._debate_reference_facts(speaker, "greatest Russian author")
    assert "Crime and Punishment" in facts or "Brothers Karamazov" in facts


def test_fallback_debate_turn_rotates_distinct_lines():
    speaker = {"name": "Tolstoy Advocate", "profile": ""}
    first = gta._fallback_debate_turn(speaker, [])
    second = gta._fallback_debate_turn(
        speaker,
        [{"speaker_name": "Tolstoy Advocate", "content": first}],
    )
    assert first != second
    assert "Tolstoy" in first
    assert not gta._has_author_mixups(first)


def test_turn_speaker_name_strips_backend_label():
    item = {
        "speaker": "Tolstoy Advocate [Local: checkpoints/fe-lora-30m on mlx-community/Qwen2.5-Coder-7B]",
        "content": "Hello",
    }
    assert gta._turn_speaker_name(item) == "Tolstoy Advocate"


def test_completed_chat_turns_skips_empty_and_typing_cursor():
    state = [
        {"speaker_name": "Human", "content": "Topic?"},
        {"speaker_name": "Alice", "content": "typing ▌"},
        {"speaker_name": "Bob", "content": ""},
        {"speaker_name": "Alice", "content": "Real answer here with enough length."},
    ]
    completed = gta._completed_chat_turns(state)
    assert len(completed) == 2
    assert completed[-1]["speaker_name"] == "Alice"


def test_hydrate_loop_chat_state_prefers_longer_store_transcript():
    store = gta._empty_conversations_store()
    conv = gta._get_active_conversation(store)
    conv["chat_state"] = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Alice", "content": "First turn with substance."},
    ]
    hydrated = gta._hydrate_loop_chat_state(store, [])
    assert len(hydrated) == 2


def test_build_debate_turn_request_includes_opponent_quote():
    speaker = {"name": "Tolstoy Advocate", "profile": "Argue Tolstoy is greatest."}
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {"speaker_name": "Dostoevsky Advocate", "content": "Crime and Punishment proves Dostoevsky is greatest Russian author."},
    ]
    req = gta._build_debate_turn_request(
        speaker,
        chat_state,
        max_tokens=128,
        temp=0.5,
        opponents=["Tolstoy Advocate", "Dostoevsky Advocate"],
    )
    user = req.messages[-1].content
    assert "Latest opposing argument (Dostoevsky Advocate)" in user
    assert "Crime and Punishment" in user
    assert "Points already made" in user


def test_turn_quality_ok_rejects_placeholder_meta():
    speaker = {"name": "Tolstoy Advocate", "profile": ""}
    assert not gta._turn_quality_ok(speaker, "Sure, here's my next turn:", [])


def test_turn_quality_ok_rejects_stand_by_meta_filler_on_first_turn():
    speaker = {"name": "Researcher", "profile": "Role: researcher."}
    chat_state = [{"speaker_name": "Human", "content": "Debate who is the greatest Russian author."}]
    filler = "I stand by my position with evidence from the reference facts above."
    assert gta._is_meta_filler_turn(filler)
    assert not gta._turn_quality_ok(speaker, filler, chat_state)
    content, ended, explicit_pass = gta._parse_debate_model_output(
        speaker, filler, chat_state, ["Researcher", "Risk Auditor"],
    )
    assert content is None
    assert explicit_pass
    assert not ended


def test_is_substantive_less_turn_skips_near_duplicate_rebuttal():
    speaker = {"name": "Researcher", "profile": "Role: researcher."}
    prior = [
        {"speaker_name": "Human", "content": "Who is the greatest Russian author?"},
        {
            "speaker_name": "Researcher",
            "content": "NEW EVIDENCE: War and Peace spans five families. CLAIM: Tolstoy is Russia's greatest novelist.",
        },
    ]
    repeat = "War and Peace spans five families and Tolstoy remains Russia's greatest novelist in my view."
    assert gta._is_substantive_less_turn(repeat, repeat, speaker, prior, phase="rebuttal")


def test_stream_signals_pass_detects_token_mid_stream():
    assert gta._stream_signals_pass("Still thinking… [[PASS]]")
    assert not gta._stream_signals_pass("I have a new argument about Anna Karenina.")


def test_resolve_lobby_room_state_prefers_live_state_then_store():
    store = gta._empty_conversations_store()
    conv = gta._get_active_conversation(store)
    conv["room_state"] = [_speaker("Stored")]
    store["conversations"][conv["id"]] = conv
    assert gta._resolve_lobby_room_state([], store)[0]["name"] == "Stored"
    assert gta._resolve_lobby_room_state([_speaker("Live")], store)[0]["name"] == "Live"


def test_upsert_room_speaker_adds_and_replaces():
    roster, replaced = gta._upsert_room_speaker([], _speaker("Alice"))
    assert not replaced
    assert [s["name"] for s in roster] == ["Alice"]
    updated = {**_speaker("Alice"), "profile": "Updated profile text."}
    roster, replaced = gta._upsert_room_speaker(roster, updated)
    assert replaced
    assert roster[0]["profile"] == "Updated profile text."


def test_append_archetype_speaker_avoids_name_collisions():
    preset = gta._find_archetype("Researcher")
    roster, unique, kind = gta._append_archetype_speaker(
        [_speaker("Researcher")],
        preset,
        local_adapter="checkpoints/fe-lora-30m",
        frontier_model="",
    )
    assert unique == "Researcher 2"
    assert kind == "debater"
    assert len(roster) == 2


def test_format_pass_turn_status_includes_debug_snippet():
    status = gta._format_pass_turn_status("Researcher", debug="[[PASS]] internal trace")
    assert "Researcher" in status
    assert "[[PASS]]" in status


def test_build_debate_turn_request_opening_for_researcher_first_turn():
    speaker = {"name": "Researcher", "profile": "Role: researcher. Evidence-first."}
    chat_state = [
        {"speaker_name": "Human", "content": "Debate who is the greatest Russian author."},
    ]
    req = gta._build_debate_turn_request(
        speaker,
        chat_state,
        max_tokens=128,
        temp=0.5,
        opponents=["Researcher", "Risk Auditor"],
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "Proposal" in system or "State your thesis clearly" in system
    assert "State your thesis clearly" in system or "State your position clearly" in system
    assert "NEW EVIDENCE + CLAIM" in user
    assert "no rebuttal" in user.lower()
    assert "Latest opposing argument" not in user
    assert "REBUTTAL → NEW EVIDENCE → CLAIM" not in user


def test_debate_turn_mode_maps_six_phases():
    assert gta._debate_turn_mode(1, 6) == "proposal"
    assert gta._debate_turn_mode(3, 6) == "develop"
    assert gta._debate_turn_mode(5, 6) == "develop"
    assert gta._debate_turn_mode(6, 6) == "conclusion"


def test_debate_phase_for_round_maps_three_phases():
    assert gta._debate_phase_for_round(1, 3) == "opening"
    assert gta._debate_phase_for_round(2, 3) == "rebuttal"
    assert gta._debate_phase_for_round(3, 3) == "final"
    assert gta._debate_phase_for_round(4, 5) == "rebuttal"
    assert gta._debate_phase_for_round(5, 5) == "final"


def test_debate_phase_for_round_single_and_double():
    assert gta._debate_phase_for_round(1, 1) == "final"
    assert gta._debate_phase_for_round(1, 2) == "opening"
    assert gta._debate_phase_for_round(2, 2) == "final"


def test_debate_pacing_context_includes_phase_count():
    ctx = gta._debate_pacing_context(3, 6, "Alice", [])
    assert "phase 3 of 6" in ctx["pacing_line"].lower()
    assert ctx["turn_mode"] == "develop"
    assert ctx["rounds_remaining"] == 3


def test_format_debate_phase_status_label():
    status = gta._format_debate_phase_status(2, 3, detail="Alice is typing…")
    assert status == "Phase 2/3: Deepen — Alice is typing…"
    assert gta._format_debate_phase_status(1, 6) == "Phase 1/6: Proposal"
    assert gta._format_debate_phase_status(6, 6) == "Phase 6/6: Conclusion"


def test_build_phase_debate_request_opening_prompt():
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    req = gta._build_phase_debate_request(
        speaker, [], 128, 0.5, opponents=["Alice", "Bob"], phase="opening",
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "State your thesis clearly" in system or "State your position clearly" in system
    assert "NEW EVIDENCE + CLAIM" in system
    assert "No rebuttal yet" in system
    assert "NEW EVIDENCE + CLAIM only" in user


def test_build_phase_debate_request_opening_second_speaker_challenges_prior():
    speaker = {"name": "Bob", "profile": "Role: researcher. Evidence-first."}
    chat_state = [
        {"speaker_name": "Human", "content": "Who is the greatest Russian author?"},
        {
            "speaker_name": "Alice",
            "content": (
                "NEW EVIDENCE: War and Peace tracks five families through Napoleon's invasion. "
                "CLAIM: Tolstoy is Russia's greatest novelist."
            ),
        },
    ]
    req = gta._build_phase_debate_request(
        speaker, chat_state, 128, 0.5, opponents=["Alice", "Bob"], phase="opening",
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "CHALLENGE + NEW EVIDENCE + CLAIM" in system
    assert "Challenge that point directly" in system
    assert "The previous speaker (Alice) argued:" in user
    assert "War and Peace" in user
    assert "CHALLENGE → NEW EVIDENCE → CLAIM" in user
    assert "no rebuttal yet" not in user.lower()
    assert "Do NOT include REBUTTAL" not in system


def test_build_phase_debate_request_opening_skips_challenge_when_profile_specifies():
    speaker = {
        "name": "Bob",
        "profile": "Role: skeptic. Challenge assumptions and respond to prior points directly.",
    }
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Alice", "content": "Alice argues Tolstoy is greatest with War and Peace evidence."},
    ]
    req = gta._build_phase_debate_request(
        speaker, chat_state, 128, 0.5, opponents=["Alice", "Bob"], phase="opening",
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "No rebuttal yet" in system
    assert "The previous speaker (Alice) argued:" not in user
    assert "CHALLENGE + NEW EVIDENCE + CLAIM" not in system


def test_prior_debater_in_phase_opening_returns_latest_debater():
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Alice", "content": "Alice opening."},
        {"speaker_name": "Bob", "content": "Bob opening."},
    ]
    name, content = gta._prior_debater_in_phase(chat_state, "Carol", "opening")
    assert name == "Bob"
    assert content == "Bob opening."


def test_opening_phase_challenge_turn_true_for_second_debater():
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Alice", "content": "Alice opening with enough length to count."},
    ]
    assert gta._opening_phase_challenge_turn(
        chat_state, "Bob", "Role: researcher.", phase="opening",
    )


def test_build_phase_debate_request_rebuttal_prompt():
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Bob", "content": "Bob argues Dostoevsky is greatest with Crime and Punishment evidence."},
    ]
    req = gta._build_phase_debate_request(
        speaker, chat_state, 128, 0.5, opponents=["Alice", "Bob"], phase="rebuttal",
        round_idx=2, max_rounds=3,
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "RESPOND/REBUT" in system
    assert "NEW EVIDENCE" in system
    assert "phase 2 of 3" in system.lower()
    assert "Latest opposing argument (Bob)" in user
    assert "RESPOND/REBUT or NEW EVIDENCE" in user or "develop turn" in user.lower()


def test_build_phase_debate_request_develop_mid_debate_six_phases():
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    chat_state = [
        {"speaker_name": "Human", "content": "Who is the greatest Russian author?"},
        {"speaker_name": "Bob", "content": "Bob argues Dostoevsky with Crime and Punishment."},
    ]
    req = gta._build_phase_debate_request(
        speaker, chat_state, 128, 0.5, opponents=["Alice", "Bob"], phase="develop",
        round_idx=3, max_rounds=6,
    )
    system = req.messages[0].content
    assert "phase 3 of 6" in system.lower()
    assert "RESPOND/REBUT" in system
    assert "Choose ONE primary move" in system


def test_build_phase_debate_request_final_prompt():
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    req = gta._build_phase_debate_request(
        speaker,
        [{"speaker_name": "Alice", "content": "Prior opening with enough length to count."}],
        128,
        0.5,
        opponents=["Alice", "Bob"],
        phase="final",
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "FINAL CLAIM" in system
    assert "without repeating verbatim" in system or "Do not recycle opening wording" in system
    assert "closing statement" in user.lower()


def test_build_adjudicator_request_structure():
    adjudicator = {"name": "Impartial Judge", "profile": "Role: impartial judge."}
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {"speaker_name": "Alice", "content": "CLAIM: Tolstoy for panoramic realism in War and Peace."},
        {"speaker_name": "Bob", "content": "CLAIM: Dostoevsky for psychological depth in Karamazov."},
    ]
    req = gta._build_adjudicator_request(adjudicator, chat_state, 256, 0.0)
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "impartial adjudicator" in system.lower()
    assert "Opening paragraph" in system
    assert "The basis of" in system
    assert "In summary" in system
    assert "narrative summary" in user.lower()
    assert "Alice:" in user
    assert "Bob:" in user
    assert "SUMMARY —" not in system
    assert "VERDICT —" not in system


def test_format_debate_summary_panel_prefixes_heading():
    room = [_speaker("Alice"), _speaker("Impartial Judge", is_judge=True), _speaker("Bob")]
    adj = gta._resolve_adjudicator(room)
    assert adj is not None
    assert adj["name"] == "Impartial Judge"
    assert gta._resolve_adjudicator([_speaker("Alice"), _speaker("Bob")]) is None


def test_default_conversation_fields_use_three_debate_phases():
    defaults = gta._default_conversation_fields()
    assert defaults["max_rounds"] == gta.DEFAULT_DEBATE_PHASES == 3


def test_stand_by_string_absent_from_emit_paths():
    assert "I stand by my position" not in "\n".join(gta.GENERIC_DEBATE_OPENING_FALLBACKS)
    for turns in gta.DEBATE_FALLBACK_TURNS.values():
        for line in turns:
            assert "I stand by my position" not in line
            assert "reference facts above" not in line
            assert "my position still stands" not in line


def test_fallback_rotates_for_renamed_archetype_by_stance():
    speaker = {
        "name": "Researcher",
        "profile": "You argue Leo Tolstoy is the greatest Russian author.",
    }
    first = gta._fallback_debate_turn(speaker, [])
    second = gta._fallback_debate_turn(
        speaker,
        [{"speaker_name": "Researcher", "content": first}],
    )
    assert first != second
    assert "Tolstoy" in first or "War and Peace" in first or "Anna Karenina" in first


def test_debate_search_enabled_respects_env(monkeypatch):
    monkeypatch.setenv("FE_DEBATE_SEARCH", "0")
    assert not gta._debate_search_enabled()
    monkeypatch.setenv("FE_DEBATE_SEARCH", "1")
    assert gta._debate_search_enabled()


def test_archetype_registry_is_wellformed():
    names = gta._archetype_names()
    assert "Researcher" in names
    assert len(names) == len(set(names)), "archetype names must be unique"
    for preset in gta.CHAT_ARCHETYPE_PRESETS:
        assert preset.get("name") and preset.get("profile")
        assert preset.get("backend") in {"Local", "Frontier"}
        assert isinstance(preset.get("is_judge", False), bool)
    # Exactly one judge preset ships by default.
    judges = [p for p in gta.CHAT_ARCHETYPE_PRESETS if p.get("is_judge")]
    assert [j["name"] for j in judges] == ["Impartial Judge"]


def test_find_archetype_is_case_insensitive_and_none_safe():
    assert gta._find_archetype("researcher")["name"] == "Researcher"
    assert gta._find_archetype("  Risk Auditor  ")["name"] == "Risk Auditor"
    assert gta._find_archetype("does-not-exist") is None
    assert gta._find_archetype(None) is None


def test_archetype_picker_includes_custom_agent_option():
    choices = gta._archetype_picker_choices()
    assert choices[-1] == gta.CUSTOM_ARCHETYPE_LABEL
    assert "Researcher" in choices
    assert len(choices) == len(gta._archetype_names()) + 1


def test_lobby_custom_fields_only_for_custom_pick():
    assert not gta._lobby_custom_fields_visible("Researcher")
    assert gta._lobby_custom_fields_visible(gta.CUSTOM_ARCHETYPE_LABEL)
    assert gta._is_custom_archetype_pick(gta.CUSTOM_ARCHETYPE_LABEL)
    assert not gta._is_custom_archetype_pick("Researcher")


def test_archetype_preview_text_for_preset_and_custom():
    text = gta._archetype_preview_text("Researcher")
    assert "Researcher" in text
    assert "evidence" in text.lower()
    custom = gta._archetype_preview_text(gta.CUSTOM_ARCHETYPE_LABEL)
    assert "custom agent" in custom.lower()


def test_load_into_form_switches_to_custom_and_prefills():
    loaded = gta._archetype_load_into_form_values("Risk Auditor")
    assert loaded is not None
    picker, name, backend, profile, is_judge = loaded
    assert picker == gta.CUSTOM_ARCHETYPE_LABEL
    assert name == "Risk Auditor"
    assert backend == "Local"
    assert "risk auditor" in profile.lower()
    assert is_judge is False
    assert gta._archetype_load_into_form_values(gta.CUSTOM_ARCHETYPE_LABEL) is None


def test_unique_speaker_name_avoids_collisions():
    assert gta._unique_speaker_name([], "Researcher") == "Researcher"
    assert gta._unique_speaker_name(["Researcher"], "Researcher") == "Researcher 2"
    assert gta._unique_speaker_name(["Researcher", "Researcher 2"], "Researcher") == "Researcher 3"
    assert gta._unique_speaker_name(["Bob"], "") == "Speaker"


def test_judge_checkbox_state_preselects_all_judges():
    room = [_speaker("Alice"), _speaker("Referee", is_judge=True), _speaker("Bob", is_judge=True)]
    choices, selected = gta._judge_checkbox_state(room)
    assert choices == ["Referee", "Bob"]
    assert selected == ["Referee", "Bob"]


def test_impartial_judge_archetype_is_flagged():
    preset = gta._find_archetype("Impartial Judge")
    assert preset is not None
    assert preset.get("is_judge") is True


def test_collect_conclude_votes_tracks_debaters_only():
    chat = [
        {"speaker_name": "Alice", "content": "Still debating.", "ended_debate": False},
        {"speaker_name": "Bob", "content": "I agree.\n[[DEBATE_CONCLUDED]]", "ended_debate": True},
        {"speaker_name": "Human", "content": "[[DEBATE_CONCLUDED]]"},
    ]
    votes = gta._collect_conclude_votes(chat, ["Alice", "Bob"])
    assert votes == {"Bob"}


def test_early_stop_reached_modes():
    names = ["Alice", "Bob"]
    assert gta._early_stop_reached(set(), names, mode="first_signal", just_signaled=True, just_voter="Alice")[0]
    assert not gta._early_stop_reached({"Alice"}, names, mode="unanimous")[0]
    assert gta._early_stop_reached({"Alice", "Bob"}, names, mode="unanimous")[0]
    three = ["Alice", "Bob", "Carol"]
    assert gta._early_stop_reached({"Alice", "Bob"}, three, mode="majority")[0]


def test_lobby_room_visibility_toggles():
    assert gta._lobby_room_visibility(False) == (True, False)
    assert gta._lobby_room_visibility(True) == (False, True)


def test_conversation_store_round_trip(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", index)
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    gta._save_conversations_store(store)
    loaded = gta._load_conversations_store()
    assert loaded["active_id"] in loaded["conversations"]
    conv_id = loaded["active_id"]
    gta._update_active_conversation(loaded, in_room=True, max_rounds=4)
    again = gta._load_conversations_store()
    assert again["conversations"][conv_id]["in_room"] is True
    assert again["conversations"][conv_id]["max_rounds"] == 4


def test_derive_conversation_title_from_human_message():
    conv = gta._create_conversation_record()
    conv["chat_state"] = [{"speaker_name": "Human", "content": "Who is the greatest Russian author?"}]
    assert "Russian author" in gta._derive_conversation_title(conv)


def test_add_conversation_creates_new_active(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", tmp_path / "index.json")
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    first_id = store["active_id"]
    gta._add_conversation(store, title="Architecture debate")
    assert store["active_id"] != first_id
    assert len(store["conversations"]) == 2
    assert store["conversations"][store["active_id"]]["title"] == "Architecture debate"


def test_conversation_list_choices_show_title_not_id():
    store = gta._empty_conversations_store()
    conv = gta._get_active_conversation(store)
    conv["chat_state"] = [{"speaker_name": "Human", "content": "debate russian authors"}]
    conv["title"] = gta._derive_conversation_title(conv)
    store["conversations"][conv["id"]] = conv
    choices = gta._conversation_list_choices(store)
    assert len(choices) == 1
    label, value = choices[0]
    assert value == conv["id"]
    assert "russian authors" in label.lower()
    assert not label.startswith("conv_")


def test_rename_conversation_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", tmp_path / "index.json")
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    gta._rename_active_conversation(store, "My custom debate title")
    conv_id = store["active_id"]
    loaded = gta._load_conversations_store()
    assert loaded["conversations"][conv_id]["title"] == "My custom debate title"
    assert loaded["conversations"][conv_id]["title_manual"] is True


def test_delete_conversation_removes_and_switches_active(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", tmp_path / "index.json")
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    first_id = store["active_id"]
    gta._add_conversation(store, title="Second chat")
    second_id = store["active_id"]
    gta._delete_conversation(store, second_id)
    assert second_id not in store["conversations"]
    assert store["active_id"] == first_id
    loaded = gta._load_conversations_store()
    assert second_id not in loaded["conversations"]


def test_delete_last_conversation_creates_fresh_one(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", tmp_path / "index.json")
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    old_id = store["active_id"]
    gta._delete_conversation(store, old_id)
    assert len(store["conversations"]) == 1
    assert store["active_id"] != old_id
    assert store["active_id"] in store["conversations"]


def test_manual_title_not_overwritten_by_new_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_INDEX", tmp_path / "index.json")
    monkeypatch.setattr(gta, "MODEL_CHAT_CONVERSATIONS_DIR", tmp_path)
    store = gta._empty_conversations_store()
    gta._rename_active_conversation(store, "Pinned title")
    gta._update_active_conversation(
        store,
        chat_state=[{"speaker_name": "Human", "content": "totally different opening message"}],
    )
    conv = gta._get_active_conversation(store)
    assert conv["title"] == "Pinned title"


def test_conversation_title_truncates_long_opening():
    conv = gta._create_conversation_record()
    long_msg = "word " * 40
    conv["chat_state"] = [{"speaker_name": "Human", "content": long_msg.strip()}]
    title = gta._derive_conversation_title(conv)
    assert len(title) <= gta.CONVERSATION_TITLE_MAX_LEN + 1
    assert title.endswith("…")


def test_conversation_in_room_reflects_active_conversation():
    store = gta._empty_conversations_store()
    assert not gta._conversation_in_room(store)
    conv = gta._get_active_conversation(store)
    conv["in_room"] = True
    store["conversations"][conv["id"]] = conv
    assert gta._conversation_in_room(store)


def test_evidence_already_used_extracts_works_and_authors():
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {
            "speaker_name": "Alice",
            "content": (
                "NEW EVIDENCE: War and Peace tracks five families through Napoleon's invasion. "
                "CLAIM: Tolstoy is Russia's greatest novelist."
            ),
        },
        {
            "speaker_name": "Bob",
            "content": "Dostoevsky's Crime and Punishment proves psychological depth beats panorama.",
        },
    ]
    used = gta._evidence_already_used(chat_state)
    assert "War and Peace" in used
    assert any("Tolstoy" in item for item in used)
    assert "Crime and Punishment" in used
    assert any("Dostoevsky" in item for item in used)


def test_build_debate_turn_request_includes_search_and_no_repeat_instructions(monkeypatch):
    monkeypatch.setenv("FE_DEBATE_SEARCH", "1")
    speaker = {"name": "Tolstoy Advocate", "profile": "Argue Tolstoy is greatest."}
    chat_state = [
        {"speaker_name": "Human", "content": "Who is greatest?"},
        {
            "speaker_name": "Dostoevsky Advocate",
            "content": "Crime and Punishment proves Dostoevsky is greatest Russian author.",
        },
    ]
    req = gta._build_debate_turn_request(
        speaker,
        chat_state,
        max_tokens=128,
        temp=0.5,
        opponents=["Tolstoy Advocate", "Dostoevsky Advocate"],
        phase="rebuttal",
    )
    system = req.messages[0].content
    user = req.messages[-1].content
    assert "reference facts from search" in system.lower()
    assert "at most once" in system.lower()
    assert "Evidence already used" in user
    assert "Crime and Punishment" in user
    assert "opponent not address" in user.lower()


def test_build_phase_debate_request_anti_repeat_pushes_new_evidence(monkeypatch):
    monkeypatch.setenv("FE_DEBATE_SEARCH", "1")
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Bob", "content": "Bob cites War and Peace for Tolstoy."},
    ]
    req = gta._build_phase_debate_request(
        speaker,
        chat_state,
        128,
        0.5,
        opponents=["Alice", "Bob"],
        phase="rebuttal",
        anti_repeat=True,
        reference_facts="- Anna Karenina is Tolstoy's other masterwork.",
    )
    user = req.messages[-1].content
    assert "NEW evidence NOT in the used list" in user
    assert "War and Peace" in user


def test_debate_fresh_reference_facts_uses_alternate_query(monkeypatch):
    monkeypatch.setenv("FE_DEBATE_SEARCH", "1")
    speaker = {"name": "Tolstoy Advocate", "profile": "Argue Tolstoy is greatest."}
    facts = gta._debate_fresh_reference_facts(
        speaker,
        "greatest Russian author",
        ["War and Peace", "Anna Karenina"],
    )
    assert facts.startswith("- ") or facts == ""


def test_build_phase_debate_request_rebuttal_escalates_depth():
    speaker = {"name": "Alice", "profile": "Role: researcher."}
    chat_state = [
        {"speaker_name": "Human", "content": "Topic"},
        {"speaker_name": "Bob", "content": "Bob argues Dostoevsky is greatest with Crime and Punishment evidence."},
    ]
    req = gta._build_phase_debate_request(
        speaker, chat_state, 128, 0.5, opponents=["Alice", "Bob"], phase="rebuttal",
    )
    system = req.messages[0].content
    assert "push the conversation deeper" in system.lower() or "deeper than the prior phase" in system.lower()
    assert "RESPOND/REBUT" in system
