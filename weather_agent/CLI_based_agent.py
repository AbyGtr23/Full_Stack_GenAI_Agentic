import json
import os
import shutil
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq

# Windows terminals default to a codepage (e.g. cp1252) that can't encode some
# characters (emoji from wttr.in, non-ASCII file contents), which would crash print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set GROQ_API_KEY in your environment or in a .env file before running this script."
    )

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"
MAX_ITERATIONS = 15
MAX_OUTPUT_CHARS = 20_000
MAX_LIST_ENTRIES = 200

# "This directory" = the repo root (parent of weather_agent/). The agent may also
# reach the repo's parent directory, so WORKSPACE_ROOT is the outer sandbox boundary.
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

# Secrets and bulky/internal folders the agent must never read, write or list.
BLOCKED_NAMES = {".git", ".env", "udenv", "hf_token.txt"}


def is_blocked_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in BLOCKED_NAMES or lowered.startswith(".env")


def resolve_safe(path: str) -> Path:
    """Resolve `path` (relative paths are relative to REPO_ROOT) and refuse
    anything outside WORKSPACE_ROOT or touching a blocked name. resolve()
    collapses `..` and follows symlinks, so neither can be used to escape."""
    raw = Path(path)
    full = (raw if raw.is_absolute() else REPO_ROOT / raw).resolve()
    try:
        relative = full.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise PermissionError(f"'{path}' is outside the allowed workspace ({WORKSPACE_ROOT}).")
    if any(is_blocked_name(part) for part in relative.parts):
        raise PermissionError(f"'{path}' is protected and cannot be accessed.")
    return full


def confirm(message: str) -> bool:
    answer = input(f"\n[confirm] {message} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def display(path: Path) -> str:
    return str(path.relative_to(WORKSPACE_ROOT)) if path != WORKSPACE_ROOT else "."


def truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text) - MAX_OUTPUT_CHARS} more characters]"


def get_weather_info(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=3"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        return f"Failed to retrieve weather information for {city}: {exc}"
    if response.status_code == 200:
        return f"Weather information for {city}: {response.text.strip()}"
    return f"Failed to retrieve weather information for {city}. Please try again later."


def list_directory(path: str = ".", recursive: bool = False) -> str:
    target = resolve_safe(path)
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    entries = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(d for d in dirnames if not is_blocked_name(d))
            base = Path(dirpath)
            for d in dirnames:
                entries.append(f"{(base / d).relative_to(target).as_posix()}/")
            for f in sorted(filenames):
                if not is_blocked_name(f):
                    entries.append((base / f).relative_to(target).as_posix())
            if len(entries) > MAX_LIST_ENTRIES:
                break
    else:
        for child in sorted(target.iterdir()):
            if not is_blocked_name(child.name):
                entries.append(child.name + ("/" if child.is_dir() else ""))

    if not entries:
        return f"{display(target)} is empty."
    shown = entries[:MAX_LIST_ENTRIES]
    result = "\n".join(shown)
    if len(entries) > MAX_LIST_ENTRIES:
        result += f"\n... [listing capped at {MAX_LIST_ENTRIES} entries]"
    return result


def read_file(path: str) -> str:
    target = resolve_safe(path)
    if not target.is_file():
        return f"Error: '{path}' is not a file."
    return truncate(target.read_text(encoding="utf-8", errors="replace"))


def write_file(path: str, content: str) -> str:
    target = resolve_safe(path)
    if target.is_dir():
        return f"Error: '{path}' is a directory."
    if target.exists() and not confirm(f"Overwrite existing file {display(target)}?"):
        return "Cancelled: the user declined to overwrite the file."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {display(target)}."


def append_file(path: str, content: str) -> str:
    target = resolve_safe(path)
    if target.is_dir():
        return f"Error: '{path}' is a directory."
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} characters to {display(target)}."


def create_directory(path: str) -> str:
    target = resolve_safe(path)
    target.mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {display(target)}."


def move_path(source: str, destination: str) -> str:
    src = resolve_safe(source)
    dst = resolve_safe(destination)
    if not src.exists():
        return f"Error: '{source}' does not exist."
    if dst.is_dir():
        dst = dst / src.name
    if dst.exists() and not confirm(f"Replace existing {display(dst)}?"):
        return "Cancelled: the user declined to replace the destination."
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {display(src)} -> {display(dst)}."


def copy_path(source: str, destination: str) -> str:
    src = resolve_safe(source)
    dst = resolve_safe(destination)
    if not src.exists():
        return f"Error: '{source}' does not exist."
    if dst.is_dir():
        dst = dst / src.name
    if dst.exists() and not confirm(f"Overwrite existing {display(dst)}?"):
        return "Cancelled: the user declined to overwrite the destination."
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return f"Copied {display(src)} -> {display(dst)}."


def delete_path(path: str, recursive: bool = False) -> str:
    target = resolve_safe(path)
    if target == REPO_ROOT or target in REPO_ROOT.parents:
        return "Error: refusing to delete the repo root or the workspace root."
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_dir() and any(target.iterdir()) and not recursive:
        return f"Error: {display(target)} is not empty. Pass recursive=true to delete it and its contents."
    if not confirm(f"Delete {display(target)}{' and everything in it' if target.is_dir() else ''}?"):
        return "Cancelled: the user declined the deletion."
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return f"Deleted {display(target)}."


available_tool = {
    "get_weather_info": get_weather_info,
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "create_directory": create_directory,
    "move_path": move_path,
    "copy_path": copy_path,
    "delete_path": delete_path,
}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_PATH = {"type": "string", "description": "Path relative to the repo root, or absolute within the workspace."}

tools = [
    _tool(
        "get_weather_info",
        "Get the current weather summary for a given city.",
        {"city": {"type": "string", "description": "City name, e.g. 'Sambalpur' or 'New Delhi'."}},
        ["city"],
    ),
    _tool(
        "list_directory",
        "List the files and folders in a directory. Folders end with '/'.",
        {"path": _PATH, "recursive": {"type": "boolean", "description": "List nested contents too. Default false."}},
        [],
    ),
    _tool("read_file", "Read a UTF-8 text file and return its contents.", {"path": _PATH}, ["path"]),
    _tool(
        "write_file",
        "Create a file with the given content, creating parent folders as needed. Asks the user before overwriting.",
        {"path": _PATH, "content": {"type": "string", "description": "Full file contents."}},
        ["path", "content"],
    ),
    _tool(
        "append_file",
        "Append text to the end of a file, creating it if missing.",
        {"path": _PATH, "content": {"type": "string", "description": "Text to append."}},
        ["path", "content"],
    ),
    _tool("create_directory", "Create a directory (and parents) if it does not exist.", {"path": _PATH}, ["path"]),
    _tool(
        "move_path",
        "Move or rename a file or directory. Asks the user before replacing an existing destination.",
        {"source": _PATH, "destination": _PATH},
        ["source", "destination"],
    ),
    _tool(
        "copy_path",
        "Copy a file or directory. Asks the user before overwriting an existing destination.",
        {"source": _PATH, "destination": _PATH},
        ["source", "destination"],
    ),
    _tool(
        "delete_path",
        "Delete a file or directory. Always asks the user to confirm first.",
        {
            "path": _PATH,
            "recursive": {"type": "boolean", "description": "Required to delete a non-empty directory."},
        },
        ["path"],
    ),
]

SYSTEM_PROMPT = f"""You are a CLI assistant that can create and manipulate files and directories using the provided tools.

Sandbox rules:
- Relative paths resolve against the repo root: {REPO_ROOT}
- You may also reach the repo's parent directory ({WORKSPACE_ROOT}), e.g. with "..". Nothing outside it is accessible.
- Secrets and internal folders (.git, .env files, HF_Token.txt, udenv) are blocked; do not try to access them.
- Overwrites, replacements and deletions are confirmed by the user at runtime. If a tool reports the user declined, do not retry it.

Work step by step: inspect with list_directory/read_file before changing things you are unsure about, and make the changes with the tools rather than only describing them. You can also look up current weather with get_weather_info. When finished, give a short, direct summary of what you did."""


def run_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return "Error: tool arguments were not valid JSON."

    if name not in available_tool:
        return f"Error: tool '{name}' does not exist. Available tools: {list(available_tool)}."

    print(f"  -> {name}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
    try:
        result = available_tool[name](**args)
    except Exception as exc:
        result = f"Error: {type(exc).__name__}: {exc}"
    return truncate(str(result))


def run_agent(message_history: list) -> None:
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=message_history,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_message = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        message_history.append(assistant_message)

        if not tool_calls:
            print(f"\nAgent: {message.content}\n")
            return

        for tool_call in tool_calls:
            message_history.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": run_tool(tool_call)}
            )

    print(f"\nStopped after {MAX_ITERATIONS} steps without reaching a final answer.\n")


def main() -> None:
    print(f"File agent ready. Workspace: {WORKSPACE_ROOT}")
    print(f"Relative paths start at: {REPO_ROOT}")
    print("Type 'exit' to quit.\n")

    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        message_history.append({"role": "user", "content": user_query})
        try:
            run_agent(message_history)
        except Exception as exc:
            print(f"\nRequest failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
