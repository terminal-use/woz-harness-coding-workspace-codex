#!/usr/bin/env python3
"""
Slack Tools - Send messages, upload files, and create canvases.

Usage:
    python slack_tools.py message --channel CHANNEL --text "Hello world"
    python slack_tools.py upload --channel CHANNEL --file /path/to/file
    python slack_tools.py canvas --title "Title" --content "# Markdown content"
    python slack_tools.py channels  # List available channels

Requires:
    - SLACK_BOT_TOKEN environment variable
    - pip install slack_sdk

Bot token scopes needed:
    - chat:write (send messages)
    - chat:write.public (post to public channels without joining)
    - files:write (upload files)
    - canvases:write (create/edit canvases)
    - channels:read (list channels)
"""

import argparse
import json
import os
import sys

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    print("Error: slack_sdk not installed. Run: pip install slack_sdk", file=sys.stderr)
    sys.exit(1)


def get_client() -> WebClient:
    """Get authenticated Slack client."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("Error: SLACK_BOT_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)
    return WebClient(token=token)


def format_response(data: dict) -> str:
    """Format response as JSON."""
    return json.dumps(data, indent=2)


# --- Messages ---

def send_message(channel: str, text: str, thread_ts: str = None,
                 blocks: str = None, unfurl_links: bool = True) -> dict:
    """Send a message to a channel."""
    client = get_client()

    kwargs = {
        "channel": channel,
        "text": text,
        "unfurl_links": unfurl_links,
    }

    if thread_ts:
        kwargs["thread_ts"] = thread_ts

    if blocks:
        try:
            kwargs["blocks"] = json.loads(blocks)
        except json.JSONDecodeError:
            print("Warning: Invalid blocks JSON, sending text only", file=sys.stderr)

    try:
        response = client.chat_postMessage(**kwargs)
        return {
            "success": True,
            "channel": response["channel"],
            "ts": response["ts"],
            "message": response["message"]["text"],
        }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response["error"],
            "detail": str(e),
        }


def send_reply(channel: str, thread_ts: str, text: str, broadcast: bool = False) -> dict:
    """Reply to a message thread."""
    client = get_client()

    try:
        response = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=text,
            reply_broadcast=broadcast,
        )
        return {
            "success": True,
            "channel": response["channel"],
            "ts": response["ts"],
            "thread_ts": thread_ts,
        }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response["error"],
        }


# --- Files ---

def upload_file(channel: str, file_path: str, title: str = None,
                initial_comment: str = None, thread_ts: str = None) -> dict:
    """Upload a file to a channel."""
    client = get_client()

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }

    kwargs = {
        "channel": channel,
        "file": file_path,
    }

    if title:
        kwargs["title"] = title
    if initial_comment:
        kwargs["initial_comment"] = initial_comment
    if thread_ts:
        kwargs["thread_ts"] = thread_ts

    try:
        # Use files_upload_v2 (the newer API)
        response = client.files_upload_v2(**kwargs)
        file_info = response.get("file", {})
        return {
            "success": True,
            "file_id": file_info.get("id"),
            "name": file_info.get("name"),
            "url": file_info.get("permalink"),
            "size": file_info.get("size"),
        }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response["error"],
            "detail": str(e),
        }


def upload_content(channel: str, content: str, filename: str,
                   filetype: str = None, title: str = None) -> dict:
    """Upload text content as a file."""
    client = get_client()

    kwargs = {
        "channel": channel,
        "content": content,
        "filename": filename,
    }

    if filetype:
        kwargs["filetype"] = filetype
    if title:
        kwargs["title"] = title

    try:
        response = client.files_upload_v2(**kwargs)
        file_info = response.get("file", {})
        return {
            "success": True,
            "file_id": file_info.get("id"),
            "name": file_info.get("name"),
            "url": file_info.get("permalink"),
        }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response["error"],
        }


# --- Canvases ---

def create_canvas(title: str = None, markdown: str = None, channel_id: str = None) -> dict:
    """Create a new standalone canvas."""
    client = get_client()

    kwargs = {}

    if title:
        kwargs["title"] = title

    if markdown:
        kwargs["document_content"] = {
            "type": "markdown",
            "markdown": markdown,
        }

    try:
        response = client.api_call("canvases.create", json=kwargs)
        if response.get("ok"):
            return {
                "success": True,
                "canvas_id": response.get("canvas_id"),
            }
        else:
            return {
                "success": False,
                "error": response.get("error"),
            }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response.get("error", str(e)),
        }


def create_channel_canvas(channel_id: str, markdown: str = None) -> dict:
    """Create a canvas attached to a channel."""
    client = get_client()

    kwargs = {
        "channel_id": channel_id,
    }

    if markdown:
        kwargs["document_content"] = {
            "type": "markdown",
            "markdown": markdown,
        }

    try:
        response = client.api_call("conversations.canvases.create", json=kwargs)
        if response.get("ok"):
            return {
                "success": True,
                "canvas_id": response.get("canvas_id"),
            }
        else:
            return {
                "success": False,
                "error": response.get("error"),
            }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response.get("error", str(e)),
        }


def edit_canvas(canvas_id: str, markdown: str, operation: str = "replace") -> dict:
    """Edit an existing canvas."""
    client = get_client()

    # For replace operation, we replace all content
    changes = [{
        "operation": operation,
        "document_content": {
            "type": "markdown",
            "markdown": markdown,
        }
    }]

    try:
        response = client.api_call("canvases.edit", json={
            "canvas_id": canvas_id,
            "changes": changes,
        })
        if response.get("ok"):
            return {
                "success": True,
                "canvas_id": canvas_id,
            }
        else:
            return {
                "success": False,
                "error": response.get("error"),
            }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response.get("error", str(e)),
        }


# --- Channels ---

def list_channels(limit: int = 100, types: str = "public_channel") -> dict:
    """List available channels."""
    client = get_client()

    try:
        response = client.conversations_list(
            limit=limit,
            types=types,
        )
        channels = []
        for ch in response.get("channels", []):
            channels.append({
                "id": ch.get("id"),
                "name": ch.get("name"),
                "is_private": ch.get("is_private", False),
                "num_members": ch.get("num_members", 0),
                "topic": ch.get("topic", {}).get("value", ""),
            })
        return {
            "success": True,
            "count": len(channels),
            "channels": channels,
        }
    except SlackApiError as e:
        return {
            "success": False,
            "error": e.response["error"],
        }


def get_channel_id(channel_name: str) -> dict:
    """Get channel ID from channel name."""
    result = list_channels(limit=500)
    if not result["success"]:
        return result

    # Remove # prefix if present
    channel_name = channel_name.lstrip("#")

    for ch in result["channels"]:
        if ch["name"] == channel_name:
            return {
                "success": True,
                "channel_id": ch["id"],
                "channel_name": ch["name"],
            }

    return {
        "success": False,
        "error": f"Channel '{channel_name}' not found",
    }


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Slack Tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Message command
    msg_p = subparsers.add_parser("message", help="Send a message")
    msg_p.add_argument("--channel", "-c", required=True, help="Channel ID or name")
    msg_p.add_argument("--text", "-t", required=True, help="Message text")
    msg_p.add_argument("--thread", help="Thread timestamp to reply to")
    msg_p.add_argument("--blocks", help="Block Kit JSON")
    msg_p.add_argument("--no-unfurl", action="store_true", help="Disable link unfurling")

    # Reply command
    reply_p = subparsers.add_parser("reply", help="Reply to a thread")
    reply_p.add_argument("--channel", "-c", required=True, help="Channel ID")
    reply_p.add_argument("--thread", "-t", required=True, help="Thread timestamp")
    reply_p.add_argument("--text", required=True, help="Reply text")
    reply_p.add_argument("--broadcast", action="store_true", help="Also post to channel")

    # Upload command
    upload_p = subparsers.add_parser("upload", help="Upload a file")
    upload_p.add_argument("--channel", "-c", required=True, help="Channel ID")
    upload_p.add_argument("--file", "-f", required=True, help="File path")
    upload_p.add_argument("--title", help="File title")
    upload_p.add_argument("--comment", help="Initial comment")
    upload_p.add_argument("--thread", help="Thread timestamp")

    # Upload content command
    content_p = subparsers.add_parser("upload-content", help="Upload text as file")
    content_p.add_argument("--channel", "-c", required=True, help="Channel ID")
    content_p.add_argument("--content", required=True, help="Text content")
    content_p.add_argument("--filename", "-f", required=True, help="Filename")
    content_p.add_argument("--filetype", help="File type (e.g., python, markdown)")
    content_p.add_argument("--title", help="File title")

    # Canvas command
    canvas_p = subparsers.add_parser("canvas", help="Create a canvas")
    canvas_p.add_argument("--title", "-t", help="Canvas title")
    canvas_p.add_argument("--content", "-c", help="Markdown content")
    canvas_p.add_argument("--file", "-f", help="Read content from file")
    canvas_p.add_argument("--channel", help="Create as channel canvas (channel ID)")

    # Edit canvas command
    edit_p = subparsers.add_parser("edit-canvas", help="Edit a canvas")
    edit_p.add_argument("--canvas-id", "-i", required=True, help="Canvas ID")
    edit_p.add_argument("--content", "-c", help="New markdown content")
    edit_p.add_argument("--file", "-f", help="Read content from file")

    # Channels command
    ch_p = subparsers.add_parser("channels", help="List channels")
    ch_p.add_argument("--limit", "-n", type=int, default=100)
    ch_p.add_argument("--private", action="store_true", help="Include private channels")

    # Get channel ID command
    chid_p = subparsers.add_parser("channel-id", help="Get channel ID from name")
    chid_p.add_argument("--name", "-n", required=True, help="Channel name")

    args = parser.parse_args()

    if args.command == "message":
        result = send_message(
            args.channel,
            args.text,
            thread_ts=args.thread,
            blocks=args.blocks,
            unfurl_links=not args.no_unfurl,
        )

    elif args.command == "reply":
        result = send_reply(args.channel, args.thread, args.text, args.broadcast)

    elif args.command == "upload":
        result = upload_file(
            args.channel,
            args.file,
            title=args.title,
            initial_comment=args.comment,
            thread_ts=args.thread,
        )

    elif args.command == "upload-content":
        result = upload_content(
            args.channel,
            args.content,
            args.filename,
            filetype=args.filetype,
            title=args.title,
        )

    elif args.command == "canvas":
        content = args.content
        if args.file:
            with open(args.file, "r") as f:
                content = f.read()

        if args.channel:
            result = create_channel_canvas(args.channel, markdown=content)
        else:
            result = create_canvas(title=args.title, markdown=content)

    elif args.command == "edit-canvas":
        content = args.content
        if args.file:
            with open(args.file, "r") as f:
                content = f.read()

        if not content:
            print("Error: --content or --file required", file=sys.stderr)
            sys.exit(1)

        result = edit_canvas(args.canvas_id, content)

    elif args.command == "channels":
        types = "public_channel,private_channel" if args.private else "public_channel"
        result = list_channels(args.limit, types)

    elif args.command == "channel-id":
        result = get_channel_id(args.name)

    print(format_response(result))

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
