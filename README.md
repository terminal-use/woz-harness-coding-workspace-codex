# Coding Workspace Codex Harness Template

This template provisions a repo-cloning coding workspace harness that uses Codex as the runtime assistant.

## What it does

- Clones `repo_url` into `/workspace` on task create.
- Bootstraps GitHub auth (`gh`) when `github_token` is provided.
- Reuses Codex thread state across events for iterative coding workflows.
- Supports practical PR/CI loops inside the cloned repo.

## Runtime stack

- `sdk_type: codex_agent_sdk`
- Python runtime with `openai-agents`
- Codex CLI (`@openai/codex`) installed in image
- `git`, `gh`, `bubblewrap`, `node`, `npm` available in container

## Expected task params

- `repo_url` (required)
- `repo_owner` (optional)
- `repo_name` (optional)
- `github_token` (optional, recommended for private repos)
- `github_login` (optional)

## Notes

- Bundled skills are copied into `/workspace/.codex/skills`.
- The bootstrap flow renders `{{HARNESS_AGENT_NAME}}` in `config.yaml` during provisioning.
