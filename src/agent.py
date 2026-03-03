"""Coding Workspace Codex Agent.

References:
- https://developers.openai.com/codex/multi-agent
- https://developers.openai.com/codex/mcp
- https://developers.openai.com/codex/guides/agents-md
- https://developers.openai.com/codex/config-basic
- https://developers.openai.com/codex/config-advanced
- https://developers.openai.com/codex/config-reference
- https://developers.openai.com/codex/config-sample
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from terminaluse.lib import AgentServer, TaskContext, make_logger
from terminaluse.types import Event, TextPart

from .helpers import (
    WORKSPACE_DIR,
    build_authenticated_clone_url,
    configure_git_identity,
    configure_runtime_logging,
    git_env,
    redact_secret,
    run_cmd,
    task_param_str,
    wait_for_workspace_ready,
    workspace_ready,
)

configure_runtime_logging()
logger = make_logger(__name__)

os.environ.setdefault("CODEX_HOME", "/app/codex")
CODEX_EXEC_TIMEOUT_SECONDS = 1800

server = AgentServer()


def _state_thread_id(state: Any) -> str | None:
    """Read persisted thread_id from task state."""
    if not isinstance(state, dict):
        return None
    value = state.get("thread_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_codex_jsonl(stdout: str) -> tuple[str | None, str | None]:
    """Extract thread_id and final assistant text from `codex exec --json` output."""
    thread_id: str | None = None
    last_message: str | None = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and candidate.strip():
                thread_id = candidate.strip()

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue

        text = item.get("text")
        if isinstance(text, str) and text.strip():
            last_message = text.strip()

    return thread_id, last_message


def _run_codex_cli(
    *,
    prompt: str,
    model: str | None,
    thread_id: str | None,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run Codex CLI in non-interactive mode, optionally resuming a thread."""
    model_args = ["-m", model] if model else []
    if thread_id:
        args = [
            "codex",
            "exec",
            "resume",
            "--json",
            *model_args,
            thread_id,
            prompt,
        ]
    else:
        args = [
            "codex",
            "exec",
            "--json",
            *model_args,
            prompt,
        ]

    return run_cmd(
        args,
        cwd=WORKSPACE_DIR,
        timeout=CODEX_EXEC_TIMEOUT_SECONDS,
        env=env,
    )


def _error_text(result: subprocess.CompletedProcess[str], token: str | None) -> str:
    """Build a user-facing error message from Codex CLI subprocess output."""
    parts: list[str] = []
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()

    if stderr:
        parts.append(stderr)
    if stdout and not parts:
        parts.append(stdout)

    message = "\n".join(parts).strip() or "Codex CLI failed without output."
    return redact_secret(message, token)[:8000]


@server.on_create
async def handle_create(ctx: TaskContext, params: dict[str, Any]):
    """Clone repo and set up workspace."""
    repo_url = params.get("repo_url")
    github_token = params.get("github_token")
    github_login = params.get("github_login")
    git_author_email = params.get("git_author_email")
    logger.info(
        "task_create task_id=%s repo_url=%s has_token=%s",
        ctx.task.id,
        repo_url,
        bool(github_token),
    )

    await ctx.state.create(state={"workspace_ready": False, "thread_id": None})

    if not repo_url:
        logger.error("missing_repo_url task_id=%s", ctx.task.id)
        return

    if not github_token and isinstance(repo_url, str) and repo_url.startswith("https://github.com/"):
        logger.warning("no_github_token task_id=%s repo_url=%s", ctx.task.id, repo_url)

    try:
        clone_url, used_embedded_token = build_authenticated_clone_url(
            str(repo_url), github_token if isinstance(github_token, str) else None
        )
        clone_cmd = ["git"]
        if not github_token:
            clone_cmd.extend(["-c", "credential.helper="])
        clone_cmd.extend(["clone", "--depth", "1", clone_url, WORKSPACE_DIR])

        result = run_cmd(clone_cmd, timeout=300, env=git_env())
        if result.returncode != 0:
            stderr = (result.stderr or "").strip() or "Unknown git clone failure"
            stderr = redact_secret(stderr, github_token if isinstance(github_token, str) else None)
            if not github_token and "could not read Username" in stderr:
                stderr = "Repository may be private. Reconnect GitHub and retry."
            logger.warning("clone_failed task_id=%s reason=%s", ctx.task.id, stderr)
            return

        if used_embedded_token:
            run_cmd(["git", "remote", "set-url", "origin", str(repo_url)], cwd=WORKSPACE_DIR)

        configure_git_identity(
            github_login if isinstance(github_login, str) else None,
            git_author_email if isinstance(git_author_email, str) else None,
        )
        if github_token:
            os.environ["GH_TOKEN"] = str(github_token)
            os.environ["GITHUB_TOKEN"] = str(github_token)

        await ctx.state.update({"workspace_ready": True, "thread_id": None})
        await ctx.messages.send("Workspace is ready.")

    except subprocess.TimeoutExpired:
        logger.warning("clone_timeout task_id=%s", ctx.task.id)
    except Exception as exc:
        logger.exception("clone_error task_id=%s error=%s", ctx.task.id, exc)


@server.on_event
async def handle_event(ctx: TaskContext, event: Event):
    """Handle user messages through Codex CLI (`codex exec`)."""
    try:
        if not isinstance(event.content, TextPart):
            await ctx.messages.send("Only text messages supported.")
            return

        user_message = event.content.text
        logger.info("event task_id=%s chars=%s", ctx.task.id, len(user_message))

        state = await ctx.state.get()
        prior_thread_id = _state_thread_id(state)
        workspace_ready_flag = bool(state.get("workspace_ready")) if isinstance(state, dict) else False

        task_github_token = task_param_str(ctx, "github_token")
        task_github_login = task_param_str(ctx, "github_login")
        task_git_author_email = task_param_str(ctx, "git_author_email")
        env = git_env()
        if task_github_token:
            env["GH_TOKEN"] = task_github_token
            env["GITHUB_TOKEN"] = task_github_token
        if task_git_author_email:
            env["GIT_AUTHOR_EMAIL"] = task_git_author_email
            env["GIT_COMMITTER_EMAIL"] = task_git_author_email

        if not workspace_ready_flag or not workspace_ready():
            ready = await wait_for_workspace_ready()
            if not ready:
                await ctx.messages.send("Workspace still initializing. Retry soon.")
                return
            await ctx.state.update({"workspace_ready": True})

        model = os.getenv("CODEX_MODEL", "").strip() or None
        configure_git_identity(task_github_login, task_git_author_email)

        result = _run_codex_cli(
            prompt=user_message,
            model=model,
            thread_id=prior_thread_id,
            env=env,
        )

        if result.returncode != 0 and prior_thread_id:
            logger.warning(
                "resume_failed task_id=%s thread_id=%s status=%s",
                ctx.task.id,
                prior_thread_id,
                result.returncode,
            )
            await ctx.state.update({"thread_id": None})
            result = _run_codex_cli(
                prompt=user_message,
                model=model,
                thread_id=None,
                env=env,
            )

        if result.returncode != 0:
            logger.warning("codex_exec_failed task_id=%s status=%s", ctx.task.id, result.returncode)
            await ctx.messages.send(_error_text(result, task_github_token))
            return

        resolved_thread_id, output_text = _parse_codex_jsonl(result.stdout or "")
        if resolved_thread_id and resolved_thread_id != prior_thread_id:
            await ctx.state.update({"thread_id": resolved_thread_id})
            logger.info("thread_id_updated task_id=%s thread_id=%s", ctx.task.id, resolved_thread_id)

        if output_text:
            await ctx.messages.send(output_text)
        else:
            await ctx.messages.send("Codex completed without a text response.")

    except Exception as exc:
        logger.exception("event_error task_id=%s error=%s", ctx.task.id, exc)
        await ctx.messages.send(str(exc))


@server.on_cancel
async def handle_cancel(ctx: TaskContext):
    """Handle task cancellation."""
    logger.info("cancelled task_id=%s", ctx.task.id)
