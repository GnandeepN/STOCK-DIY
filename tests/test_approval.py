from __future__ import annotations

import argparse
import pytest
from ai_trading_bot.core.notify import request_approval


def test_placeholder():
    """A simple placeholder test to confirm pytest is working."""
    assert True


@pytest.mark.skip(reason="This is an interactive test that requires manual approval on Telegram.")
def test_interactive_approval() -> int:
    p = argparse.ArgumentParser(description="Send a test approval to Telegram and print the result")
    p.add_argument("--title", default="Approval Test")
    p.add_argument("--text", default="This is a test approval. Click Approve or Reject.")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    ok = request_approval(title=args.title, text=args.text, timeout=args.timeout)
    print("Approved" if ok else "Rejected or timed out")
    assert ok is not None  # Simple assertion for the interactive test
    return 0


if __name__ == "__main__":
    # To run the interactive test manually:
    # python tests/test_approval.py
    raise SystemExit(test_interactive_approval())

