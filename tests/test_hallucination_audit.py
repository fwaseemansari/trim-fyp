"""Tests for evaluation/hallucination_audit.py (no network, no models)."""

import json

import pandas as pd
import pytest

from evaluation.hallucination_audit import (
    LABELS, build_audit_sheet, response_in_context, stratified_estimate, summarize_audit,
)


def _samples(n=10):
    return [{"id": f"s{i}", "question": f"q{i}?", "context": f"Fact {i} is alpha{i}. Filler one. Filler two.",
             "answer": f"alpha{i}"} for i in range(n)]


def _results(samples, scorer="bi_encoder", method="selective", level=0.5, lost=()):
    rows = []
    for i, s in enumerate(samples):
        rows.append({
            "sample_id": s["id"], "question": s["question"],
            "compression_method": method, "compression_scorer": scorer,
            "compression_window_size": 2, "compression_level": level, "backend": "groq",
            "answer_still_present": i not in lost, "gold_answer": s["answer"],
            "llm_response": s["answer"], "f1": 1.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def setup(tmp_path):
    samples = _samples()
    sp = tmp_path / "samples.json"
    sp.write_text(json.dumps(samples), encoding="utf-8")
    return tmp_path, samples, sp


def fake_compress(config, context, query):
    return context.split(". ")[0] + "."          # keep the first sentence only


def test_sheet_is_blind_and_key_maps_back(setup):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples).to_csv(p, index=False)
    sheet, key = build_audit_sheet([p], sp, n_per_file=5, out_path=str(tmp / "sheet.csv"),
                                   compress_fn=fake_compress)
    assert len(sheet) == 5
    assert not {"method", "scorer", "level", "sample_id"} & set(sheet.columns)
    assert {"method", "scorer", "level", "sample_id"} <= set(key.columns)
    assert list(sheet.row_id) == list(key.row_id)
    assert (tmp / "sheet_key.csv").exists()


def test_lost_rows_are_oversampled(setup):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples, lost={0, 1}).to_csv(p, index=False)   # only 2 of 10 lost the answer
    _, key = build_audit_sheet([p], sp, n_per_file=4, out_path=str(tmp / "s.csv"),
                               compress_fn=fake_compress)
    # both lost rows are always sampled, however the random fill falls
    assert {"s0", "s1"} <= set(key.sample_id)


def test_same_seed_same_sheet(setup):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples).to_csv(p, index=False)
    a, _ = build_audit_sheet([p], sp, 5, 7, str(tmp / "a.csv"), fake_compress)
    b, _ = build_audit_sheet([p], sp, 5, 7, str(tmp / "b.csv"), fake_compress)
    assert a.question.tolist() == b.question.tolist()


def test_recompression_mismatch_is_detected(setup, capsys):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples).to_csv(p, index=False)      # CSV says every answer survived
    # fake compressor keeps only the first sentence, which still holds the answer
    build_audit_sheet([p], sp, 4, out_path=str(tmp / "s.csv"), compress_fn=fake_compress)
    assert "reproduced answer_still_present for 4/4" in capsys.readouterr().out
    # a compressor that drops everything contradicts the CSV flags
    build_audit_sheet([p], sp, 4, out_path=str(tmp / "s2.csv"), compress_fn=lambda c, x, q: "nothing")
    assert "warning" in capsys.readouterr().out


def test_baseline_uses_original_context_and_llm_runs_are_refused(setup):
    tmp, samples, sp = setup
    base = _results(samples, method="none", scorer="none", level=1.0)
    base.to_csv(tmp / "base.csv", index=False)
    sheet, _ = build_audit_sheet([tmp / "base.csv"], sp, 3, out_path=str(tmp / "b.csv"),
                                 compress_fn=lambda *a: pytest.fail("must not compress"))
    assert sheet.compressed_context.str.contains("Filler two").all()

    llm = _results(samples, method="llm")
    llm.to_csv(tmp / "llm.csv", index=False)
    with pytest.raises(NotImplementedError):
        build_audit_sheet([tmp / "llm.csv"], sp, 3, out_path=str(tmp / "l.csv"))
    # ...unless the run saved its compressed text
    llm["compressed_context"] = "saved text"
    llm.to_csv(tmp / "llm.csv", index=False)
    sheet, _ = build_audit_sheet([tmp / "llm.csv"], sp, 3, out_path=str(tmp / "l.csv"))
    assert (sheet.compressed_context == "saved text").all()


def test_summary_counts_rates_and_validation(setup):
    tmp, samples, sp = setup
    a, b = tmp / "a.csv", tmp / "b.csv"
    _results(samples, scorer="cross_encoder").to_csv(a, index=False)
    _results(samples, scorer="tfidf").to_csv(b, index=False)
    sheet, key = build_audit_sheet([a, b], sp, 4, out_path=str(tmp / "s.csv"), compress_fn=fake_compress)

    sheet["label"] = "faithful"
    ce = key.index[key.scorer == "cross_encoder"][0]
    sheet.loc[ce, "label"] = "hallucinated"
    sheet.loc[key.index[key.scorer == "tfidf"][0], "label"] = "Unsupported_Correct "   # case/space tolerant
    sheet.to_csv(tmp / "s.csv", index=False)

    out = summarize_audit(str(tmp / "s.csv"))
    ce_row = out[out.scorer == "cross_encoder"].iloc[0]
    tf_row = out[out.scorer == "tfidf"].iloc[0]
    assert ce_row.hallucinated == 1 and ce_row.n == 4 and ce_row.hallucinated_rate == 0.25
    assert tf_row.unsupported_correct == 1 and tf_row.unsupported_rate == 0.25

    sheet.loc[0, "label"] = "maybe"
    sheet.to_csv(tmp / "s.csv", index=False)
    with pytest.raises(ValueError, match="unknown labels"):
        summarize_audit(str(tmp / "s.csv"))


def test_unlabeled_rows_are_excluded_not_counted(setup):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples).to_csv(p, index=False)
    sheet, _ = build_audit_sheet([p], sp, 4, out_path=str(tmp / "s.csv"), compress_fn=fake_compress)
    sheet.loc[:1, "label"] = "faithful"       # only 2 of 4 labelled
    sheet.to_csv(tmp / "s.csv", index=False)
    assert summarize_audit(str(tmp / "s.csv")).iloc[0].n == 2


def test_helpers():
    assert response_in_context("GTE", "It was sold to GTE in 1973.")
    assert not response_in_context("", "anything")
    assert not response_in_context("Paris", "It was sold to GTE.")
    assert stratified_estimate([(300, 0, 0)]) == (0.0, 0.0, 0.0)
    est, lo, hi = stratified_estimate([(100, 10, 5)])
    assert est == 0.5 and lo < 0.5 < hi
    assert set(LABELS) == {"faithful", "unsupported_correct", "hallucinated", "unsure"}


def test_rates_are_weighted_by_stratum_not_by_sample_share(setup):
    tmp, samples, sp = setup
    p = tmp / "r.csv"
    _results(samples, lost={0, 1}).to_csv(p, index=False)   # 2 lost, 8 kept in the full run
    sheet, key = build_audit_sheet([p], sp, 4, out_path=str(tmp / "s.csv"), compress_fn=fake_compress)
    # 2 lost + 2 kept sampled; both lost answers hallucinated, both kept faithful
    sheet["label"] = ["hallucinated" if st == "lost" else "faithful" for st in key.stratum]
    sheet.to_csv(tmp / "s.csv", index=False)
    row = summarize_audit(str(tmp / "s.csv")).iloc[0]
    assert row.hallucinated == 2 and row.n == 4
    assert row.hallucinated_rate == 0.2        # 2/10 of the real run, not 2/4 of the sample