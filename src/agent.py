"""Coding Workspace Codex Agent.

Docs:
- https://developers.openai.com/codex/guides/agents-sdk/
- https://openai.github.io/openai-agents-python/tools/

Clones a GitHub repo on task creation and runs Codex for coding assistance.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.mcp import MCPServerStdio
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

os.environ.setdefault("CODEX_HOME", "/root/.codex")

DEFAULT_CODEX_MODEL = "gpt-5.3-codex"

server = AgentServer()


def _state_thread_id(state: Any) -> str | None:
    """Read thread_id from task state."""
    if not isinstance(state, dict):
        return None
    value = state.get("thread_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _find_thread_id(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> str | None:
    """Recursively search for a thread id in mixed SDK result objects."""
    if value is None or depth > 6:
        return None

    if isinstance(value, str):
        return None

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return None
    seen.add(value_id)

    if isinstance(value, dict):
        thread_id = value.get("threadId") or value.get("thread_id")
        if isinstance(thread_id, str) and thread_id.strip():
            return thread_id.strip()
        for item in value.values():
            found = _find_thread_id(item, depth=depth + 1, seen=seen)
            if found:
                return found
        return None

    if isinstance(value, (list, tuple, set)):
        for item in value:
            found = _find_thread_id(item, depth=depth + 1, seen=seen)
            if found:
                return found
        return None

    for attr in ("thread_id", "threadId"):
        if hasattr(value, attr):
            thread_id = getattr(value, attr)
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()

    for attr in ("final_output", "output", "raw_output", "content", "item", "result", "data", "payload"):
        if hasattr(value, attr):
            found = _find_thread_id(getattr(value, attr), depth=depth + 1, seen=seen)
            if found:
                return found

    if hasattr(value, "__dict__"):
        return _find_thread_id(vars(value), depth=depth + 1, seen=seen)

    return None


def _extract_thread_id_from_result(result: Any) -> str | None:
    """Extract threadId from an Agent SDK run result."""
    for attr in ("final_output", "new_items", "items", "output_items"):
        found = _find_thread_id(getattr(result, attr, None))
        if found:
            return found
    return None


def _result_text(output: Any) -> str | None:
    """Normalize run output into text for user-facing messages."""
    if isinstance(output, str):
        return output.strip() or None
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


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

        configure_git_identity(github_login if isinstance(github_login, str) else None)
        if github_token:
            os.environ["GH_TOKEN"] = str(github_token)
            os.environ["GITHUB_TOKEN"] = str(github_token)

        await ctx.state.update({"workspace_ready": True})
        await ctx.messages.send("Workspace is ready.")

    except subprocess.TimeoutExpired:
        logger.warning("clone_timeout task_id=%s", ctx.task.id)
    except Exception as exc:
        logger.exception("clone_error task_id=%s error=%s", ctx.task.id, exc)


@server.on_event
async def handle_event(ctx: TaskContext, event: Event):
    """Handle user messages via Codex MCP server."""
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
        if task_github_token:
            os.environ["GH_TOKEN"] = task_github_token
            os.environ["GITHUB_TOKEN"] = task_github_token

        if not workspace_ready_flag or not workspace_ready():
            ready = await wait_for_workspace_ready()
            if not ready:
                await ctx.messages.send("Workspace still initializing. Retry soon.")
                return
            await ctx.state.update({"workspace_ready": True})

        codex_mcp = MCPServerStdio(
            name="Codex MCP",
            params={
                "command": "codex",
                "args": [
                    "mcp-server",
                    "-c",
                    'sandbox_mode="danger-full-access"',
                    "-c",
                    'approval_policy="never"',
                ],
            },
            cache_tools_list=True,
        )

        instructions = (
            "Use Codex MCP tools for coding tasks.\n"
            "Call exactly one tool for each user message.\n"
            "If `Current thread_id` is present below, call `codex-reply` with that exact `threadId`.\n"
            "If it is absent, call `codex`.\n"
            "Always set `prompt` to the user message.\n"
        )
        if prior_thread_id:
            instructions += f"Current thread_id: {prior_thread_id}\n"
        else:
            instructions += "Current thread_id: none\n"
        if task_github_token:
            instructions += "Runtime: GH_TOKEN configured. Verify with `gh auth status`.\n"
        else:
            instructions += "Runtime: No GitHub token provided.\n"
        instructions += "Repo cloned at /workspace. Work with local files."

        async with codex_mcp:
            agent = Agent(
                name="Codex",
                instructions=instructions,
                model=os.getenv("CODEX_MODEL", "").strip() or DEFAULT_CODEX_MODEL,
                model_settings=ModelSettings(tool_choice="required"),
                mcp_servers=[codex_mcp],
            )
            result = await Runner.run(agent, user_message)

        resolved_thread_id = _extract_thread_id_from_result(result)
        if resolved_thread_id and resolved_thread_id != prior_thread_id:
            await ctx.state.update({"thread_id": resolved_thread_id})
            logger.info("thread_id_updated task_id=%s thread_id=%s", ctx.task.id, resolved_thread_id)

        output = _result_text(result.final_output)
        if output:
            await ctx.messages.send(output)
        else:
            await ctx.messages.send("Codex completed without a text response.")

    except Exception as exc:
        logger.exception("event_error task_id=%s error=%s", ctx.task.id, exc)
        await ctx.messages.send(str(exc))


@server.on_cancel
async def handle_cancel(ctx: TaskContext):
    """Handle task cancellation."""
    logger.info("cancelled task_id=%s", ctx.task.id)
