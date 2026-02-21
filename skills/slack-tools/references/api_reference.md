# Slack API Reference

## Authentication

Set `SLACK_BOT_TOKEN` environment variable with your bot's OAuth token.

## Required Scopes

| Scope | Purpose |
|-------|---------|
| `chat:write` | Send messages |
| `chat:write.public` | Post to public channels without joining |
| `files:write` | Upload files |
| `canvases:write` | Create/edit canvases |
| `channels:read` | List public channels |
| `groups:read` | List private channels |

## API Methods

### chat.postMessage

Send a message to a channel.

```python
client.chat_postMessage(
    channel="C123456",       # Channel ID
    text="Hello world",      # Message text
    thread_ts="1234.5678",   # Optional: reply in thread
    blocks=[...],            # Optional: Block Kit JSON
)
```

**Limits:** 4,000 chars recommended, 40,000 max.

### files.upload_v2

Upload a file to a channel.

```python
client.files_upload_v2(
    channel="C123456",
    file="/path/to/file",
    title="File Title",
    initial_comment="Check this out",
)
```

Or upload text content:

```python
client.files_upload_v2(
    channel="C123456",
    content="code here",
    filename="script.py",
    filetype="python",
)
```

### canvases.create

Create a standalone canvas.

```python
client.api_call("canvases.create", json={
    "title": "My Canvas",
    "document_content": {
        "type": "markdown",
        "markdown": "# Heading\n\nContent here"
    }
})
```

### conversations.canvases.create

Create a channel canvas (attached to channel tab).

```python
client.api_call("conversations.canvases.create", json={
    "channel_id": "C123456",
    "document_content": {
        "type": "markdown",
        "markdown": "# Channel Canvas"
    }
})
```

### canvases.edit

Edit an existing canvas.

```python
client.api_call("canvases.edit", json={
    "canvas_id": "F123456",
    "changes": [{
        "operation": "replace",
        "document_content": {
            "type": "markdown",
            "markdown": "# Updated content"
        }
    }]
})
```

## Canvas Markdown Support

Canvases support:
- Headings (h1-h3)
- Bold, italic, strikethrough
- Bulleted and ordered lists
- Checklists
- Code blocks
- Quote blocks
- Markdown tables
- Links
- @mentions
- Emojis

## Rate Limits

| Tier | Limit |
|------|-------|
| Tier 1 | 1 request/min |
| Tier 2 | 20 requests/min |
| Tier 3 | 50 requests/min |
| Tier 4 | 100 requests/min |

Most methods are Tier 3 or 4. Canvas methods may have stricter limits.

## Common Errors

| Error | Meaning |
|-------|---------|
| `channel_not_found` | Invalid channel ID or bot not in channel |
| `not_in_channel` | Bot needs to join the channel first |
| `missing_scope` | Token lacks required permission |
| `rate_limited` | Too many requests, wait and retry |
| `invalid_blocks` | Malformed Block Kit JSON |
