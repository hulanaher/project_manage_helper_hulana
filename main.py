import json
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

DATA_FILE = Path("projects.json")

# ---------- 儲存層 ----------
def load_projects() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def save_projects(projects: list[dict]) -> None:
    DATA_FILE.write_text(
        json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def resolve_targets(projects: list[dict], payload: dict) -> tuple[list[int], str]:
    idx_raw = payload.get("index")
    if idx_raw not in (None, ""):
        try:
            idx = int(idx_raw) - 1
        except (TypeError, ValueError):
            return [], f"Invalid index: {idx_raw}. Use the number from 'list projects'."
        if idx < 0 or idx >= len(projects):
            return [], f"Index {idx_raw} is out of range. Use 'list projects' to see valid numbers."
        return [idx], ""

    name = payload.get("project_name")
    if not name:
        return [], "Please specify the project by number (from 'list projects') or by name."

    target = str(name).strip().lower()
    matches = [
        i for i, p in enumerate(projects)
        if str(p.get("project_name", "")).strip().lower() == target
    ]
    if not matches:
        return [], f'Project "{name}" not found.'
    if len(matches) > 1:
        lines = [f'Multiple projects named "{name}". Please retry using the number:']
        for i in matches:
            p = projects[i]
            lines.append(f'  {i + 1}. {p["project_name"]} — customer: {p["customer"]}')
        return [], "\n".join(lines)
    return matches, ""

# ---------- LLM ----------
def build_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "agnes-2.5-flash"),
        base_url=os.getenv("LLM_BASE_URL", "https://apihub.agnes-ai.com/v1"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0,
    )

# ---------- LangGraph State ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    payload: dict
    pending: list[str]
    pending_confirm: bool   # 是否正在等待 clear 的二次確認
    reply: str

REQUIRED_FIELDS = ["project_name", "customer"]

ROUTER_PROMPT = """You are an intent router for a project-management console app.
Read the user's message and reply with ONLY a JSON object, no markdown.

Schema:
{
  "intent": "create" | "list" | "get" | "update" | "delete" | "clear" | "confirm" | "quit" | "unknown",
  "index": integer or null,
  "project_name": string or null,
  "customer": string or null,
  "start_date": string or null,
  "location": string or null,
  "status": string or null,
  "notes": string or null,
  "confirm_value": "yes" | "no" | null
}

Rules:
- "create": user wants to add a project.
- "list": user wants to see all projects.
- "get": user wants details of ONE specific project.
- "update": user wants to change fields of an existing project.
- "delete": user wants to remove an existing project.
- "clear": user wants to remove ALL projects (e.g. "clear all", "delete everything", "清空").
- "confirm": user is answering a yes/no question (yes/no/ok/cancel/確定/取消).
  Set "confirm_value" to "yes" or "no".
- "quit": user says quit/exit.
- "unknown": anything else.
- If the user refers to a project by its list number (e.g. "delete #2"), set "index".
- Otherwise extract "project_name" if a name is given.
- Extract fields only if explicitly present; otherwise null. Never invent values.
- Messages may be in English or Chinese.
"""

# ---------- Nodes ----------
def router_node(state: AgentState) -> AgentState:
    llm = build_llm()
    last_user = state["messages"][-1].content
    resp = llm.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(last_user)])
    raw = resp.content.strip().strip("`").removeprefix("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"intent": "unknown"}
    if not isinstance(data, dict):
        data = {"intent": "unknown"}
    payload = {k: v for k, v in data.items() if k != "intent" and v}
    return {
        "intent": data.get("intent", "unknown"),
        "payload": payload,
    }

def create_node(state: AgentState) -> AgentState:
    payload = dict(state.get("payload", {}))

    if state.get("pending"):
        llm = build_llm()
        missing = state["pending"]
        question = (
            f"Extract values for {missing} from the user text. "
            f"Reply with JSON only, keys={missing}, null if absent. "
            f"Text: {state['messages'][-1].content}"
        )
        resp = llm.invoke([HumanMessage(question)])
        raw = resp.content.strip().strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload.update({k: v for k, v in parsed.items() if v})
        except json.JSONDecodeError:
            pass

    missing = [f for f in REQUIRED_FIELDS if not payload.get(f)]
    if missing:
        return {
            "pending": missing,
            "payload": payload,
            "reply": f"我需要更多資訊：{', '.join(missing)}。",
        }

    projects = load_projects()
    projects.append({
        "project_name": payload["project_name"],
        "customer": payload["customer"],
        "start_date": payload.get("start_date"),
        "location": payload.get("location"),
        "status": payload.get("status") or "new",
        "notes": payload.get("notes"),
    })
    save_projects(projects)
    return {
        "pending": [],
        "payload": {},
        "reply": f'Done. Created project "{payload["project_name"]}" for customer {payload["customer"]}.',
    }

def list_node(state: AgentState) -> AgentState:
    projects = load_projects()
    if not projects:
        return {"reply": "You have no projects yet."}
    lines = [f"Here are your projects ({len(projects)}):"]
    for i, p in enumerate(projects, 1):
        lines.append(f'{i}. {p["project_name"]} — customer: {p["customer"]}')
    return {"reply": "\n".join(lines)}

def get_node(state: AgentState) -> AgentState:
    projects = load_projects()
    if not projects:
        return {"reply": "You have no projects yet."}
    indices, err = resolve_targets(projects, state.get("payload", {}))
    if err:
        return {"reply": err}

    p = projects[indices[0]]
    lines = [f'Project "{p["project_name"]}":']
    for key in ["customer", "start_date", "location", "status", "notes"]:
        lines.append(f"  {key}: {p.get(key)}")
    return {"reply": "\n".join(lines)}

def update_node(state: AgentState) -> AgentState:
    projects = load_projects()
    if not projects:
        return {"reply": "You have no projects yet."}
    indices, err = resolve_targets(projects, state.get("payload", {}))
    if err:
        return {"reply": err}

    payload = state.get("payload", {})
    updatable = ["customer", "start_date", "location", "status", "notes"]
    changes = {k: payload[k] for k in updatable if payload.get(k)}
    if not changes:
        return {
            "reply": "What should I change? Provide at least one of: "
                     + ", ".join(updatable)
        }

    i = indices[0]
    projects[i].update(changes)
    save_projects(projects)
    changed = ", ".join(f"{k}={v}" for k, v in changes.items())
    return {"reply": f'Updated project "{projects[i]["project_name"]}": {changed}.'}

def delete_node(state: AgentState) -> AgentState:
    projects = load_projects()
    if not projects:
        return {"reply": "You have no projects yet."}
    indices, err = resolve_targets(projects, state.get("payload", {}))
    if err:
        return {"reply": err}

    i = indices[0]
    removed = projects.pop(i)
    save_projects(projects)
    return {"reply": f'Deleted project "{removed["project_name"]}".'}

def clear_node(state: AgentState) -> AgentState:
    projects = load_projects()
    if not projects:
        return {"reply": "There are no projects to clear.", "pending_confirm": False}
    if not state.get("pending_confirm"):
        return {
            "pending_confirm": True,
            "reply": (
                f"This will delete ALL {len(projects)} project(s). "
                "Type 'yes' to confirm, or 'no' to cancel."
            ),
        }

    return {
        "pending_confirm": True,
        "reply": "Please answer 'yes' to confirm clearing all projects, or 'no' to cancel.",
    }

def confirm_node(state: AgentState) -> AgentState:
    if not state.get("pending_confirm"):
        return {"reply": "Nothing to confirm.", "pending_confirm": False}

    payload = state.get("payload", {})
    answer = str(payload.get("confirm_value", "")).lower()

    if answer == "yes":
        save_projects([])
        return {"reply": "All projects cleared.", "pending_confirm": False}
    if answer == "no":
        return {"reply": "Cancelled. Nothing was deleted.", "pending_confirm": False}

    return {
        "reply": "Please answer 'yes' or 'no'.",
        "pending_confirm": True,
    }

def quit_node(state: AgentState) -> AgentState:
    return {"reply": "Bye."}

def unknown_node(state: AgentState) -> AgentState:
    return {
        "reply": (
            "Sorry, I didn't understand. Try one of:\n"
            "  create a project called <name> for customer <customer>\n"
            "    optional: starts <start_date>, location <location>, status <status>, notes \"<notes>\"\n"
            "  list projects\n"
            "  get <number>\n"
            "  update <number> <field> to <value>   (fields: customer, start_date, location, status, notes)\n"
            "  delete <number>\n"
            "  clear\n"
            "  quit"
        )
    }

# ---------- Routing ----------
def route(state: AgentState) -> str:
    if state.get("intent") == "quit":
        return "quit"
    if state.get("pending_confirm"):
        # 等待確認時，只接受 confirm；其他 intent 一律回 clear_node 提醒
        if state.get("intent") == "confirm":
            return "confirm"
        return "clear"
    if state.get("pending"):
        return "create"
    return state.get("intent", "unknown")

# ---------- Build graph ----------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("create", create_node)
    g.add_node("list", list_node)
    g.add_node("get", get_node)
    g.add_node("update", update_node)
    g.add_node("delete", delete_node)
    g.add_node("clear", clear_node)
    g.add_node("confirm", confirm_node)
    g.add_node("quit", quit_node)
    g.add_node("unknown", unknown_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route, {
        "create": "create",
        "list": "list",
        "get": "get",
        "update": "update",
        "delete": "delete",
        "clear": "clear",
        "confirm": "confirm",
        "quit": "quit",
        "unknown": "unknown",
    })
    for node in ["create", "list", "get", "update", "delete", "clear", "confirm", "quit", "unknown"]:
        g.add_edge(node, END)
    return g.compile()

# ---------- REPL ----------
def main():
    app = build_graph()
    state: AgentState = {
        "messages": [],
        "intent": "",
        "payload": {},
        "pending": [],
        "pending_confirm": False,
        "reply": "",
    }
    print(
        "Project Agent ready.\n"
        "Try commands like:\n"
        "  create a project called Alpha for customer Acme\n"
        "  create project Beta for Globex, starts 2026-01-15, location Taipei, status active, notes \"kickoff next week\"\n"
        "    (optional fields: start_date, location, status, notes; status defaults to 'new')\n"
        "  list projects\n"
        "  get 1 / update 1 status to active / delete 1\n"
        "  update 1 location Taipei, notes \"moved to new office\"\n"
        "    (updatable fields: customer, start_date, location, status, notes)\n"
        "  clear            (deletes ALL projects, asks for confirmation)\n"
        "  quit             (or exit / Ctrl+C)\n"
    )
    
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue

        state["messages"] = state["messages"] + [HumanMessage(user)]
        state = {**state, **app.invoke(state)}
        print(state["reply"])
        state["messages"] = state["messages"] + [AIMessage(state["reply"])]

        if state.get("intent") == "quit":
            break
        state["intent"] = ""
        state["payload"] = {}
        # pending 只在 create 補答流程保留；pending_confirm 由 clear/confirm 自行管理
        if state.get("intent") not in ("create",) and not state.get("pending_confirm"):
            state["pending"] = []

if __name__ == "__main__":
    main()