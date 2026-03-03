# Coding Workspace Codex Harness Template

Minimal template for a repo-cloning coding workspace that runs Codex via the Codex CLI.

## BIG NOTE

**To target Codex models (including `codex-5.3-pro`), this template uses Codex CLI config files.**

- Do not rely on the SDK + MCP server path for `codex-5.3-pro`.
- Use `.codex/config.toml` and per-agent config files as the source of truth.

## What this template does

- Clones the target repository into `/workspace`.
- Optionally boots GitHub CLI auth when a token is provided.
- Ensures Codex project config files exist in `/workspace/.codex/`.
- Runs `codex exec` and `codex exec resume` (JSON mode) to preserve thread state across task events.

## Runtime

- `sdk_type: codex_agent_sdk`
- Codex CLI (`@openai/codex`) preinstalled
- `git`, `gh`, `bubblewrap`, `node`, and `npm` available in the container

## OpenAI Docs

- Multi-agent (Codex CLI): https://developers.openai.com/codex/multi-agent
- MCP with Codex CLI: https://developers.openai.com/codex/mcp
- AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- Config basics: https://developers.openai.com/codex/config-basic
- Config advanced: https://developers.openai.com/codex/config-advanced
- Config reference: https://developers.openai.com/codex/config-reference
- Config sample: https://developers.openai.com/codex/config-sample
- Agents SDK guide (background/reference): https://developers.openai.com/codex/guides/agents-sdk/

## Default Config Files Created (If Missing)

- `/workspace/.codex/config.toml`
- `/workspace/.codex/agents/coding.toml`
