"""Helper utilities for the Coding Workspace Codex Agent."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

from terminaluse.lib import TaskContext, make_logger

NOISY_RUNTIME_LOGGERS = (
    "httpx",
    "httpcore",
    "uvicorn.access",
    "terminaluse.lib.telemetry",
    "terminaluse.lib.sdk.fastacp",
    "opentelemetry",
    "opentelemetry.instrumentation",
)

WORKSPACE_DIR = "/workspace"
WORKSPACE_PATH = Path(WORKSPACE_DIR)
WORKSPACE_GIT_PATH = WORKSPACE_PATH / ".git"


def _env_log_level(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def configure_runtime_logging() -> None:
    """Reduce infra noise while keeping agent-level signal visible."""
    app_level = _env_log_level("AGENT_APP_LOG_LEVEL", logging.INFO)
    infra_level = _env_log_level("AGENT_INFRA_LOG_LEVEL", logging.WARNING)
    logging.getLogger().setLevel(app_level)
    for logger_name in NOISY_RUNTIME_LOGGERS:
        logging.getLogger(logger_name).setLevel(infra_level)


logger = make_logger(__name__)


def run_cmd(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 120,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command."""
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def git_env() -> dict[str, str]:
    """Get environment with git prompts disabled."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env


def build_authenticated_clone_url(
    repo_url: str, github_token: str | None
) -> tuple[str, bool]:
    """Return a clone URL with embedded token when cloning from GitHub HTTPS."""
    if (
        not github_token
        or not repo_url.startswith("https://github.com/")
        or "@" in repo_url.split("://", 1)[1]
    ):
        return repo_url, False
    authed = repo_url.replace(
        "https://github.com/",
        f"https://x-access-token:{github_token}@github.com/",
        1,
    )
    return authed, True


def redact_secret(text: str, secret: str | None) -> str:
    """Replace secret with *** in text."""
    if not secret:
        return text
    return text.replace(secret, "***")


def configure_git_identity(github_login: str | None) -> None:
    """Set git user.name and user.email."""
    name = github_login or "TerminalUse Agent"
    email = (
        f"{github_login}@users.noreply.github.com"
        if github_login
        else "terminaluse-agent@users.noreply.github.com"
    )
    run_cmd(["git", "config", "user.name", name], cwd=WORKSPACE_DIR)
    run_cmd(["git", "config", "user.email", email], cwd=WORKSPACE_DIR)


def task_param_str(ctx: TaskContext, key: str) -> str | None:
    """Extract a string parameter from task params."""
    params = getattr(ctx.task, "params", None)
    if not isinstance(params, dict):
        return None
    value = params.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def workspace_ready() -> bool:
    """Check if workspace has a git repository."""
    return WORKSPACE_GIT_PATH.exists()


async def wait_for_workspace_ready(
    *,
    timeout_seconds: float = 45.0,
    poll_seconds: float = 0.5,
) -> bool:
    """Wait for workspace to be ready (git repo exists)."""
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while time.monotonic() <= deadline:
        if workspace_ready():
            return True
        await asyncio.sleep(max(0.05, poll_seconds))
    return workspace_ready()
