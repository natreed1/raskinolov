#!/usr/bin/env python3
"""Probe debate quality; scores repetition and checks for on-topic Russian-author content."""
import os
import re
import sys

os.environ.setdefault("FE_DEBATE_SEARCH", "1")

sys.path.insert(0, "scripts")
import game_task_arena as g

TOPIC = "Who is the greatest Russian author — Dostoevsky, Tolstoy, or Chekhov? Defend your pick."

SPEAKERS = [
    {
        "name": "Tolstoy Advocate",
        "backend": "Local",
        "profile": (
            g._find_archetype("Implementer")["profile"]
            + " You argue Leo Tolstoy is the greatest Russian author (War and Peace, Anna Karenina)."
        ),
        "local_adapter": "checkpoints/fe-lora-30m",
    },
    {
        "name": "Dostoevsky Advocate",
        "backend": "Local",
        "profile": (
            g._find_archetype("Explorer")["profile"]
            + " You argue Fyodor Dostoevsky is the greatest Russian author (Crime and Punishment, Brothers Karamazov)."
        ),
        "local_adapter": "checkpoints/fe-lora-30m",
    },
]

RUSSIAN_MARKERS = re.compile(
    r"dostoevsk|tolstoy|chekhov|gogol|pushkin|turgenev|bulgakov|"
    r"russia|russian|novel|war and peace|karamazov|crime and punishment|"
    r"anna karenina|literature|author|writer",
    re.I,
)


def score_transcript(state):
    turns = [t for t in state if (t.get("content") or "").strip() and t.get("speaker_name") != "Human"]
    if not turns:
        return {"turns": 0, "repetitions": 0, "on_topic": 0, "unique_openers": 0, "mixups": 0, "stance_violations": 0}
    reps = mixups = stance_v = 0
    openers = set()
    on_topic = 0
    speaker_by_name = {s["name"]: s for s in SPEAKERS}
    for i, t in enumerate(turns):
        text = t["content"]
        spk = speaker_by_name.get(t["speaker_name"], {"name": t["speaker_name"], "profile": ""})
        if RUSSIAN_MARKERS.search(text):
            on_topic += 1
        if g._has_author_mixups(text):
            mixups += 1
        if g._violates_stance_lock(spk, text):
            stance_v += 1
        openers.add(text[:60].lower())
        for prev in turns[:i]:
            if g._text_similarity(text, prev["content"]) >= 0.38 or g._is_near_copy(text, prev["content"]):
                reps += 1
                break
    return {
        "turns": len(turns),
        "repetitions": reps,
        "on_topic": on_topic,
        "unique_openers": len(openers),
        "mixups": mixups,
        "stance_violations": stance_v,
    }


def run_debate(max_rounds=3, max_tokens=240, temp=0.55):
    state = [{"speaker": "Human", "speaker_name": "Human", "content": TOPIC}]
    names = [s["name"] for s in SPEAKERS]
    for rnd in range(1, max_rounds + 1):
        for speaker in SPEAKERS:
            text = g._generate_debate_turn_text(speaker, state, names, max_tokens, temp)
            state.append({
                "speaker": speaker["name"],
                "speaker_name": speaker["name"],
                "content": text,
            })
            print(f"\n--- Round {rnd} | {speaker['name']} ---\n{text}\n")
    return state


if __name__ == "__main__":
    state = run_debate()
    metrics = score_transcript(state)
    print("METRICS:", metrics)
    ok = (
        metrics["turns"] >= 4
        and metrics["repetitions"] <= 2
        and metrics["on_topic"] >= max(2, metrics["turns"] // 2)
        and metrics["unique_openers"] >= metrics["turns"] - 2
        and metrics["mixups"] <= 2
        and metrics["stance_violations"] <= 2
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
