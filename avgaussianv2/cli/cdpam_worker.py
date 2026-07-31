"""Persistent external worker for the paper-compatible CDPAM metric."""

from __future__ import annotations

from avgaussianv2.cdpam import serve_cdpam_worker


def main() -> int:
    return serve_cdpam_worker()


if __name__ == "__main__":
    raise SystemExit(main())
