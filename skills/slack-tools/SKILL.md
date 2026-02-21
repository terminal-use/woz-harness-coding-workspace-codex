---
name: slack-tools
description: Send messages, upload files, and create canvases in Slack. Use when the agent needs to post updates to Slack channels, share files with a team, create documentation in Slack canvases, or reply to threads. Requires SLACK_BOT_TOKEN and slack_sdk package.
---

# Slack Tools

Send messages, upload files, and create Slack canvases.

## Setup

1. Install the SDK: `pip install slack_sdk`
2. Set `SLACK_BOT_TOKEN` environment variable
3. Ensure bot has required scopes (see API Reference)

## Quick Start

```bash
# List channels to find channel ID
python scripts/slack_tools.py channels

# Send a message
python scripts/slack_tools.py message -c C123456 -t "Hello from the agent!"

# Upload a file
python scripts/slack_tools.py upload -c C123456 -f ./report.pdf

# Create a canvas
python scripts/slack_tools.py canvas --title "Project Notes" --content "# Overview\n\nDetails here..."
```

## Commands

| Command | Description |
|---------|-------------|
| `message -c CHANNEL -t TEXT` | Send a message |
| `reply -c CHANNEL -t THREAD -text TEXT` | Reply in a thread |
| `upload -c CHANNEL -f FILE` | Upload a file |
| `upload-content -c CHANNEL --content TEXT -f NAME` | Upload text as file |
| `canvas --title TITLE --content MD` | Create standalone canvas |
| `canvas --channel ID --content MD` | Create channel canvas |
| `edit-canvas -i ID --content MD` | Edit existing canvas |
| `channels` | List available channels |
| `channel-id -n NAME` | Get channel ID from name |

## Send Messages

```bash
# Simple message
python scripts/slack_tools.py message -c C123456 -t "Build completed successfully"

# Reply to a thread
python scripts/slack_tools.py reply -c C123456 -t 1234567890.123456 --text "Fixed in latest commit"

# Reply and broadcast to channel
python scripts/slack_tools.py reply -c C123456 -t 1234567890.123456 --text "Important update" --broadcast
```

## Upload Files

```bash
# Upload a file
python scripts/slack_tools.py upload -c C123456 -f ./report.pdf --title "Weekly Report"

# Upload with comment
python scripts/slack_tools.py upload -c C123456 -f ./data.csv --comment "Here's the data you requested"

# Upload code/text content directly
python scripts/slack_tools.py upload-content -c C123456 --content "print('hello')" -f script.py --filetype python
```

## Create Canvases

```bash
# Standalone canvas (owned by bot)
python scripts/slack_tools.py canvas --title "Project Plan" --content "# Goals\n\n- Item 1\n- Item 2"

# Channel canvas (appears in channel tab)
python scripts/slack_tools.py canvas --channel C123456 --content "# Channel Guidelines\n\nRules here..."

# Create from file
python scripts/slack_tools.py canvas --title "Documentation" --file ./docs.md

# Edit existing canvas
python scripts/slack_tools.py edit-canvas -i F123456 --content "# Updated Content"
```

## Response Format

All commands return JSON:

```json
{
  "success": true,
  "channel": "C123456",
  "ts": "1234567890.123456"
}
```

On error:
```json
{
  "success": false,
  "error": "channel_not_found"
}
```

## API Reference

See [references/api_reference.md](references/api_reference.md) for scopes, methods, and rate limits.
