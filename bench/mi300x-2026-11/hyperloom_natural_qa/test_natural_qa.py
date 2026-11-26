import json
from pathlib import Path
import natural_qa as nq


def test_token_f1_matches_longbench_rules():
    assert nq.token_f1("The Eiffel Tower", "eiffel tower") == 1.0
    assert nq.token_f1("", "x") == 0.0
    assert abs(nq.score("Paris, France", ["paris", "rome"]) - 2 / 3) < 1e-9


def test_paired_ci_detects_a_consistent_drop_and_not_noise():
    base = [0.6, 0.4, 0.8, 0.5] * 25
    d, lo, hi = nq.paired_ci(base, [b - 0.1 for b in base])
    assert abs(d + 0.1) < 1e-9 and hi < 0
    d, lo, hi = nq.paired_ci(base, list(base))
    assert d == 0 and lo <= 0 <= hi


def test_compare_flags_regression(tmp_path: Path):
    s, b, c = tmp_path / "set.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    items = [{"id": f"hotpotqa-{i:03d}", "task": "hotpotqa", "input": "x", "outputs": ["alpha beta"]} for i in range(60)]
    s.write_text("".join(json.dumps(x) + "\n" for x in items))
    b.write_text("".join(json.dumps({"id": x["id"], "text": "alpha beta"}) + "\n" for x in items))
    c.write_text("".join(json.dumps({"id": x["id"], "text": "alpha beta" if i % 3 else "gamma"}) + "\n" for i, x in enumerate(items)))
    r = nq.compare(s, b, c)["hotpotqa"]
    assert r["baseline_f1"] == 1.0 and r["candidate_f1"] < 0.7 and r["regression"] is True
