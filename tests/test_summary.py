"""Tests for experiments/summary.py (reads the committed results/ files)."""

from experiments.summary import build_summary


def test_summary_reports_key_results() -> None:
    text = build_summary()
    assert "SUMMARY OF RESULTS" in text
    assert "565.21" in text            # base-case cost
    assert "3.7892" in text            # base-case price
    assert "6-8" in text               # first congested line
    assert "Infeasible" in text


def test_summary_is_plain_ascii() -> None:
    # ASCII only, so it prints on any Windows console or redirected file.
    build_summary().encode("ascii")
