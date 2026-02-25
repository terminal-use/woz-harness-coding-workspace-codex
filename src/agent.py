"""Coding Workspace Codex Agent.

Clones a GitHub repo on task creation and runs Codex for coding assistance.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.extensions.experimental.codex import (
    CodexToolStreamEvent,
    ThreadOptions,
    codex_tool,
)
from agents.extensions.experimental.codex.events import (
    ThreadError,
    ThreadEvent,
    ThreadStartedEvent,
    TurnFailedEvent,
)
from terminaluse.lib import AgentServer, TaskContext, make_logger
from terminaluse.types import Event, TextPart

from helpers import (
    WORKSPACE_DIR,
    build_authenticated_clone_url,
    configure_git_identity,
    configure_runtime_logging,
    git_env,
    redact_secret,
    run_cmd,
    send_status,
    task_param_str,
    wait_for_workspace_ready,
    workspace_ready,
)

configure_runtime_logging()
logger = make_logger(__name__)

DEFAULT_CODEX_MODEL = "gpt-5.3-codex"

SYSTEM_PROMPT = """You are a coding assistant working in a repository at /workspace.

Core behavior:
- Read the existing code before proposing edits.
- Be concise, practical, and explicit about assumptions.
- Prefer small, safe, testable changes.
- Explain what you changed and why.

When the user asks for a "plan":
- Provide a short numbered plan first.
- Update the plan as steps complete.

When asked to get changes "PR-ready":
1. Create a branch named `codex/<short-topic>`.
2. Implement requested changes.
3. Run relevant local checks (lint/tests/build).
4. Commit with a clear message.
5. Open or update a PR using `gh`.
6. Check CI status using `gh pr checks` / `gh run`.
7. If checks fail, inspect logs, fix, push, and repeat until green or blocked.
8. End with PR URL, check status summary, and any remaining risks.

GitHub tooling:
- If `gh` is available and authenticated, use it for PRs and CI checks.
- Never print or expose tokens in command output.
"""

server = AgentServer()


@server.on_create
async def handle_create(ctx: TaskContext, params: dict[str, Any]):
    """Clone repo and set up workspace."""
    repo_url = params.get("repo_url")
    github_token = params.get("github_token")
    github_login = params.get("github_login")
    logger.info(
        "task_create task_id=%s repo_url=%s has_token=%s",
        ctx.task.id,
        repo_url,
        bool(github_token),
    )

    await ctx.state.create(state={"thread_id": None, "workspace_ready": False})

    if not repo_url:
        await send_status(ctx, "error", "Missing `repo_url` task param.")
        return

    await send_status(ctx, "cloning", f"Cloning {repo_url} ...")
    if not github_token and isinstance(repo_url, str) and repo_url.startswith("https://github.com/"):
        await send_status(ctx, "warning", "No GitHub token. Private repos may fail.")

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
            await send_status(ctx, "error", f"Clone failed: {stderr}")
            return

        if used_embedded_token:
            run_cmd(["git", "remote", "set-url", "origin", str(repo_url)], cwd=WORKSPACE_DIR)

        configure_git_identity(github_login if isinstance(github_login, str) else None)
        if github_token:
            os.environ["GH_TOKEN"] = str(github_token)
            os.environ["GITHUB_TOKEN"] = str(github_token)

        await ctx.state.update({"workspace_ready": True})
        await send_status(ctx, "ready", "Repository cloned successfully.")
        await ctx.messages.send("Workspace is ready.")

    except subprocess.TimeoutExpired:
        logger.warning("clone_timeout task_id=%s", ctx.task.id)
        await send_status(ctx, "error", "Clone timed out.")
    except Exception as exc:
        logger.exception("clone_error task_id=%s error=%s", ctx.task.id, exc)
        await send_status(ctx, "error", f"Clone error: {exc}")


@server.on_event
async def handle_event(ctx: TaskContext, event: Event):
    """Handle user messages via Codex."""
    try:
        if not isinstance(event.content, TextPart):
            await ctx.messages.send(
                TurnFailedEvent(error=ThreadError(message="Only text messages supported."))
            )
            return

        user_message = event.content.text
        logger.info("event task_id=%s chars=%s", ctx.task.id, len(user_message))

        state = await ctx.state.get()
        thread_id = state.get("thread_id") if isinstance(state, dict) else None
        if not isinstance(thread_id, str):
            thread_id = None
        workspace_ready_flag = bool(state.get("workspace_ready")) if isinstance(state, dict) else False

        task_github_token = task_param_str(ctx, "github_token")
        if task_github_token:
            os.environ["GH_TOKEN"] = task_github_token
            os.environ["GITHUB_TOKEN"] = task_github_token

        if not workspace_ready_flag or not workspace_ready():
            ready = await wait_for_workspace_ready()
            if not ready:
                await ctx.messages.send(
                    TurnFailedEvent(error=ThreadError(message="Workspace still initializing. Retry soon."))
                )
                return
            await ctx.state.update({"workspace_ready": True})

        instructions = SYSTEM_PROMPT
        if task_github_token:
            instructions += "\nRuntime: GH_TOKEN configured. Verify with `gh auth status`."
        else:
            instructions += "\nRuntime: No GitHub token provided."
        instructions += "\nRepo cloned at /workspace. Work with local files."

        os.environ.setdefault("CODEX_HOME", "/root/.codex")
        resolved_thread_id = thread_id

        async def on_stream(payload: CodexToolStreamEvent) -> None:
            nonlocal resolved_thread_id
            thread_event: ThreadEvent = payload.event
            await ctx.messages.send(thread_event)
            if isinstance(thread_event, ThreadStartedEvent):
                resolved_thread_id = thread_event.thread_id

        agent = Agent(
            name="Codex",
            instructions=instructions,
            model=os.getenv("CODEX_MODEL", "").strip() or DEFAULT_CODEX_MODEL,
            model_settings=ModelSettings(tool_choice="required"),
            tools=[
                codex_tool(
                    thread_id=thread_id,
                    default_thread_options=ThreadOptions(
                        working_directory=WORKSPACE_DIR,
                        skip_git_repo_check=True,
                        sandbox_mode="danger-full-access",
                        approval_policy="never",
                    ),
                    on_stream=on_stream,
                    failure_error_function=None,
                )
            ],
        )

        await Runner.run(agent, user_message)

        if resolved_thread_id and resolved_thread_id != thread_id:
            await ctx.state.update({"thread_id": resolved_thread_id})

    except Exception as exc:
        logger.exception("event_error task_id=%s error=%s", ctx.task.id, exc)
        await ctx.messages.send(TurnFailedEvent(error=ThreadError(message=str(exc))))


@server.on_cancel
async def handle_cancel(ctx: TaskContext):
    """Handle task cancellation."""
    logger.info("cancelled task_id=%s", ctx.task.id)
