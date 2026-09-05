"""Read-only health check for an authenticated remote vLLM endpoint."""
from __future__ import annotations

import argparse

from llm_endpoint import check_llm_health, resolve_llm_connection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--base-url", help="Optional override; REMOTE_LLM_BASE_URL is preferred")
    parser.add_argument("--timeout", type=float, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    connection = resolve_llm_connection("remote", args.base_url)
    available = check_llm_health(connection, args.model, args.timeout)
    print(
        f"Remote vLLM is healthy at {connection.base_url}; "
        f"expected model is available; served model count={len(available)}"
    )


if __name__ == "__main__":
    main()
