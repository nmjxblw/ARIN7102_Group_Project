"""Unified application entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

APP_DIR = Path(__file__).resolve().parent
STREAMLIT_APP = APP_DIR / "ui_module" / "streamlit_app.py"

# 支持的启动模式（用户要求的 cli / ui）
CLI_ALIASES = {"cli"}
UI_ALIASES = {"ui"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the terminal launcher or the Streamlit frontend.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=sorted(CLI_ALIASES | UI_ALIASES),
        help="Launch mode: cli 为命令行模式，ui 为 Web UI 模式。",
    )
    return parser


def prepare_runtime_context() -> None:
    # Keep imports and relative resource lookups aligned with the historical app/ startup mode.
    app_dir_str = str(APP_DIR)
    if app_dir_str not in sys.path:
        sys.path.insert(0, app_dir_str)
    os.chdir(APP_DIR)


def launch_cli() -> None:
    prepare_runtime_context()
    from launcher_module import app_run

    print(f"{__package__}.{__name__} 被作为主程序运行，启动 CLI 模式...")
    app_run()


def launch_streamlit(streamlit_args: Sequence[str]) -> None:
    os.chdir(APP_DIR)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP.relative_to(APP_DIR)),
        *streamlit_args,
    ]
    os.execv(sys.executable, command)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.mode in CLI_ALIASES:
        if extra_args:
            parser.error("CLI mode does not accept extra arguments.")
        launch_cli()
        return

    # ui 模式（原 streamlit 模式）
    launch_streamlit(extra_args)


if __name__ == "__main__":
    main()