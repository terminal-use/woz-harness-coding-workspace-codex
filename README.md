# Coding Workspace Codex Harness Template

Minimal template for a repo-cloning coding workspace that runs Codex via the Codex CLI.

## Note

**The SDK + MCP server path does not currently support the latest Codex models (including `codex-5.3-pro`), so this template uses Codex CLI instead.**

- Use baked `codex/config.toml` (`/app/codex/config.toml`) as the source of truth for Codex runtime settings.

## What this template does

- Clones the target repository into `/workspace`.
- Optionally boots GitHub CLI auth when a token is provided.
- Supports optional `git_author_email` task param for commit author/committer email.
- Uses baked Codex config at `/app/codex/config.toml`.
- Points Codex CLI to `/app/codex/config.toml` via `CODEX_HOME=/app/codex`.
- Runs `codex exec` and `codex exec resume` (JSON mode) to preserve thread state across task events.

## Runtime

- `sdk_type: codex_agent_sdk`
- Codex CLI (`@openai/codex`) preinstalled
- `git`, `gh`, `node`, and `npm` available in the container

## OpenAI Docs

- Multi-agent (Codex CLI): https://developers.openai.com/codex/multi-agent
- MCP with Codex CLI: https://developers.openai.com/codex/mcp
- AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- Config basics: https://developers.openai.com/codex/config-basic
- Config advanced: https://developers.openai.com/codex/config-advanced
- Config reference: https://developers.openai.com/codex/config-reference/
- Config sample: https://developers.openai.com/codex/config-sample
- Agents SDK guide (background/reference): https://developers.openai.com/codex/guides/agents-sdk/

## Codex Config

- Source in image: `codex/config.toml` (from repo path `codex/config.toml`)
- Runtime path: `/app/codex/config.toml`
