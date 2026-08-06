# Week 5 Agents Notebooks + Light Slide Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three companion notebooks to `week-5/`, wire Notebook + Colab links into `week-5/index.html`, and apply targeted Class 2–3 slide copy patches so lecture matches LangChain-from-Class-2 labs.

**Architecture:** Notebooks are built with a one-shot `nbformat` generator script per class (run once, commit the `.ipynb`, discard the generator — not a repo artifact). Class 1 uses the raw `groq` SDK only. Class 2–3 use `langchain.agents.create_agent` + `langchain_groq.ChatGroq` + `@tool`, with `InMemorySaver` for multi-turn memory in Class 3. Slide edits are text-only inside existing nodes; index cards gain action links in the existing Blueprint theme.

**Tech Stack:** Python 3, `nbformat`, `groq`, `langchain`, `langchain-groq`, `langgraph` (pulled in by modern `create_agent` / checkpointer), plain HTML/CSS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-06-week-5-agents-notebooks-design.md`
- Keep existing class titles and deck structure; no new slides; no no-code class; no LlamaIndex coded labs.
- Never hardcode API keys. Resolve `GROQ_API_KEY` from env, then Colab `userdata`, with a friendly missing-key message. Live-API cells must not crash the notebook when the key is absent — print guidance and skip/return early.
- Colab links: `https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-5/class-N.ipynb`
- Default Groq model for all live cells: `llama-3.3-70b-versatile`
- LangChain agent API to use (current docs): `from langchain.agents import create_agent` and `from langchain.tools import tool`; model via `ChatGroq(...)`. Cap loops with invoke `config={"recursion_limit": N}`.
- Challenge cells: markdown prompt + `# TODO` starter code — no full solutions.
- Mentions of LlamaIndex / LangGraph are prose only (except installing `langgraph` if required as a dependency of `create_agent` / `InMemorySaver`).

## File map

| File | Responsibility |
|---|---|
| `week-5/class-1.ipynb` | Conceptual agent demos with raw Groq |
| `week-5/class-2.ipynb` | Plan→Act→Observe via LangChain tools + Groq |
| `week-5/class-3.ipynb` | Multi-tool agent + memory + guardrails |
| `week-5/index.html` | Notebook + Colab links per class |
| `week-5/class-2.html` | Demo + challenges wording → LangChain |
| `week-5/class-3.html` | Framework guidance flip + LlamaIndex/LangGraph callout |

---

### Task 1: `week-5/class-1.ipynb` — What Makes an Agent an Agent?

**Files:**
- Create: `week-5/class-1.ipynb`

**Interfaces:**
- Produces: `ask(system_prompt, user_prompt, ...)` Groq helper; `manual_chain(steps)` that runs a list of `(system, user)` turns and prints each reply; `classify_autonomy(scenario)` student TODO.

- [ ] **Step 1: Write the notebook generator**

Save to `/tmp/build_week5_class1.py` (scratch only):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
CELLS = [
("markdown", """# Class 1 - What Makes an Agent an Agent?

**Week 5: Introduction to AI Agents**

### Learning objectives
By the end of this notebook you will be able to:
- Explain how an agent differs from a single prompt/response.
- Describe autonomy as a spectrum (low / medium / high).
- Walk a manual Plan → Act → Observe chain in plain Python + Groq.
- Spot agent-like behavior in short scenarios.

Run cells in order with **Shift+Enter**. You need a `GROQ_API_KEY` for the live demo cells (Colab secret or env var). Classification exercises run without a key."""),

("markdown", """## Setup

Install the Groq SDK, then resolve your API key. Never hardcode a key in a notebook.

```bash
export GROQ_API_KEY="gsk-..."
```

In Colab: add a secret named `GROQ_API_KEY` and enable notebook access."""),

("code", """!pip install -q groq"""),

("code", """import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    from google.colab import userdata
    GROQ_API_KEY = GROQ_API_KEY or userdata.get("GROQ_API_KEY")
except Exception:
    pass

if not GROQ_API_KEY:
    print(
        "No GROQ_API_KEY found.\\n"
        "Set it in your environment or add a Colab secret named GROQ_API_KEY.\\n"
        "Live demo cells below will skip until a key is available."
    )
else:
    print("Found GROQ_API_KEY. You're ready to run the demo cells.")


def ask(system_prompt, user_prompt, model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=400):
    \"\"\"Send a system + user message to Groq. Returns assistant text, or None if no key.\"\"\"
    if not GROQ_API_KEY:
        print("Skipping live call — no GROQ_API_KEY set.")
        return None
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content"""),

("markdown", """## 1. A single prompt vs an agent-shaped task

A single completion is one shot: question in, answer out. Agents keep going — they can decide on a next action, use a tool or intermediate step, observe the result, and continue until the goal is met.

Below we ask a multi-step question in one shot. Notice what the model invents vs what it would need to look up."""),

("code", """single_shot_question = (
    "I have a flight landing in Kathmandu at 18:00 local time. "
    "Should I pack a jacket tonight, and what is 14C in Fahrenheit?"
)

reply = ask(
    system_prompt="You are a helpful assistant. Answer briefly.",
    user_prompt=single_shot_question,
)
print(reply or "(no reply — set GROQ_API_KEY to run this cell)")"""),

("markdown", """## 2. A manual Plan → Act → Observe chain

We simulate an agent without a framework. Each step is a separate Groq call you control:

1. **Plan** — decide what information we need.
2. **Act** — "look up" weather with a fake tool (a Python dict).
3. **Observe** — feed the tool result back and ask for a final answer.

This is the same loop Class 2 will automate with LangChain tools."""),

("code", """MOCK_WEATHER = {
    "Kathmandu": {"temp_c": 14, "condition": "rain"},
    "Pokhara": {"temp_c": 18, "condition": "clear"},
}


def fake_get_weather(city: str) -> dict:
    \"\"\"Stand-in for a real weather API — the Act step.\"\"\"
    return MOCK_WEATHER.get(city, {"temp_c": None, "condition": "unknown"})


def manual_chain(city: str):
    # Plan
    plan = ask(
        system_prompt="You plan tool use. Reply with one city name to look up and nothing else.",
        user_prompt=f"User wants jacket advice for tonight near {city}. Which city should we query?",
    )
    print("PLAN:", plan)

    # Act (our code runs the tool — not the model)
    lookup_city = city  # keep deterministic for the demo
    observation = fake_get_weather(lookup_city)
    print("ACT -> fake_get_weather:", observation)

    # Observe
    final = ask(
        system_prompt="You give packing advice. Use only the weather observation provided.",
        user_prompt=(
            f"City: {lookup_city}. Observation: {observation}. "
            "Should the traveler bring a jacket? One short paragraph."
        ),
    )
    print("OBSERVE / FINAL:", final)
    return final


manual_chain("Kathmandu")"""),

("markdown", """## 3. Autonomy is a spectrum

Not every LLM app is an agent. Use these labels:

- **Low** — model replies once; human does all follow-up.
- **Medium** — model may call tools / take a few steps inside a fixed loop.
- **High** — model pursues a goal over many steps with little supervision.

Classify each scenario by writing `low`, `medium`, or `high` next to it."""),

("code", """SCENARIOS = [
    "A FAQ bot answers one customer question from a system prompt, then stops.",
    "A coding assistant edits files, runs tests, and retries until tests pass (with a human approving merges).",
    "A travel helper calls get_weather then answers whether to bring a jacket.",
    "An overnight research agent browses the web for hours and emails a report with no check-ins.",
    "Autocomplete suggests the next few tokens in an IDE as you type.",
]

# Fill in your labels: "low" | "medium" | "high"
your_labels = {
    SCENARIOS[0]: "",  # TODO
    SCENARIOS[1]: "",
    SCENARIOS[2]: "",
    SCENARIOS[3]: "",
    SCENARIOS[4]: "",
}

for s, label in your_labels.items():
    print(f"[{label or '?'}] {s}")"""),

("markdown", """## Closing

You now have a working definition: agents pursue a goal through a loop of decisions and actions. Next class, LangChain + Groq will own the Plan → Act → Observe wiring so you are not hand-rolling every step.

**Next:** Class 2 — The Agent Loop (LangChain tools)."""),

("markdown", """## Challenges

Complete each challenge in the TODO cells. Do not peek at Class 2 yet — stay on plain Groq / Python."""),

("markdown", """### Challenge 01 — Change the one-shot question
Ask a different one-sentence multi-step question with `ask(...)` and print the reply."""),

("code", """# TODO: call ask(...) with your own one-sentence question
pass"""),

("markdown", """### Challenge 02 — Write a 3-step manual chain from scratch
Pick a new fake tool (e.g. `fake_fx_rate(from_currency, to_currency)`) and run Plan → Act → Observe yourself."""),

("code", """# TODO: define a fake tool dict + manual_chain-style function, then run it
pass"""),

("markdown", """### Challenge 03 — Classify five scenarios
Finish the `your_labels` dict in Section 3 (or copy it here) so every scenario has `low`, `medium`, or `high`."""),

("code", """# TODO: paste/complete your_labels and print them
pass"""),

("markdown", """### Challenge 04 (stretch) — Spot the agent
Write two short transcripts (3–5 lines each). Mark which one is more agent-like and why (2–3 sentences)."""),

("code", """# TODO: write transcript_a, transcript_b, and a short justification string
pass"""),
]

nb.cells = [
    nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
    for kind, src in CELLS
]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
path = "/Users/praveshchapagain/Desktop/genai-foundations-b2/week-5/class-1.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print("Wrote", path)
```

- [ ] **Step 2: Generate the notebook**

Run: `python /tmp/build_week5_class1.py`  
Expected: prints `Wrote .../week-5/class-1.ipynb`

- [ ] **Step 3: Smoke-check without an API key**

Run:

```bash
cd /Users/praveshchapagain/Desktop/genai-foundations-b2
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("week-5/class-1.ipynb").read_text())
assert len(nb["cells"]) >= 12
srcs = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])
assert "GROQ_API_KEY" in srcs and "manual_chain" in srcs and "Challenges" in srcs
assert "langchain" not in srcs.lower()
print("class-1 structural OK:", len(nb["cells"]), "cells")
PY
```

Expected: `class-1 structural OK` and **no** `langchain` in the notebook.

- [ ] **Step 4: Commit**

```bash
git add week-5/class-1.ipynb
git commit -m "$(cat <<'EOF'
Add Week 5 Class 1 agents notebook.

Conceptual Groq demos for agent vs prompt, manual Plan-Act-Observe, and autonomy challenges.
EOF
)"
```

---

### Task 2: `week-5/class-2.ipynb` — The Agent Loop with LangChain

**Files:**
- Create: `week-5/class-2.ipynb`

**Interfaces:**
- Produces: `get_weather(location: str) -> str` `@tool`; `build_agent(tools, max_turns=8)` returning a `create_agent` graph; `run_agent(agent, question, thread_id="class2")` helper that invokes with `recursion_limit`.

- [ ] **Step 1: Write the notebook generator**

Save to `/tmp/build_week5_class2.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
CELLS = [
("markdown", """# Class 2 - The Agent Loop — Plan, Act, Observe

**Week 5: Introduction to AI Agents**

### Learning objectives
By the end of this notebook you will be able to:
- Define a LangChain tool with `@tool` and clear docstring descriptions.
- Run Plan → Act → Observe using `create_agent` + Groq (`ChatGroq`).
- Explain why a loop beats a one-shot answer when tools are involved.
- Cap the loop with `recursion_limit` so a stuck agent cannot burn quota.

**Stack note:** We use **LangChain** + **Groq** in class. **LlamaIndex** can express the same tool-use pattern with its own agent abstractions — pick one stack per project and stay consistent."""),

("markdown", """## Setup

```bash
export GROQ_API_KEY="gsk-..."
```

Install the minimal LangChain + Groq stack:"""),

("code", """!pip install -q langchain langchain-groq langgraph"""),

("code", """import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    from google.colab import userdata
    GROQ_API_KEY = GROQ_API_KEY or userdata.get("GROQ_API_KEY")
except Exception:
    pass

if not GROQ_API_KEY:
    print(
        "No GROQ_API_KEY found.\\n"
        "Set it in your environment or add a Colab secret named GROQ_API_KEY.\\n"
        "Agent cells will skip until a key is available."
    )
else:
    print("Found GROQ_API_KEY. LangChain + Groq demos are ready.")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq


def make_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY,
    )


def build_agent(tools, system_prompt=None, max_turns=8):
    \"\"\"Build a LangChain tool-calling agent. max_turns maps to recursion_limit on invoke.\"\"\"
    llm = make_llm()
    if llm is None:
        return None
    prompt = system_prompt or (
        "You are a careful assistant. Use tools when they help. "
        "After you have enough observations, give a final answer."
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    agent._class2_max_turns = max_turns  # stash for run_agent
    return agent


def run_agent(agent, question: str, thread_id: str = "class2"):
    if agent is None:
        print("Skipping — no GROQ_API_KEY / agent.")
        return None
    max_turns = getattr(agent, "_class2_max_turns", 8)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": max_turns},
    )
    messages = result.get("messages", [])
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", str(m))
        print(f"--- {role} ---")
        print(content)
        print()
    return result"""),

("markdown", """## 1. Define a tool

The model never executes Python itself. It **requests** a tool call; **your runtime** runs the function and returns an observation.

Write a clear docstring — vague descriptions cause wrong or missing tool calls."""),

("code", """@tool
def get_weather(location: str) -> str:
    \"\"\"Return a short weather summary for a city name like Kathmandu or Pokhara.\"\"\"
    catalog = {
        "kathmandu": "temp_c=14, condition=rain",
        "pokhara": "temp_c=18, condition=clear",
    }
    key = location.strip().lower()
    if key not in catalog:
        return f"No weather data for '{location}'. Try Kathmandu or Pokhara."
    return catalog[key]


print(get_weather.invoke({"location": "Kathmandu"}))"""),

("markdown", """## 2. Plan → Act → Observe with `create_agent`

Ask a question that needs the tool. Watch the printed message trace: the assistant plans a tool call, the tool acts, then the assistant observes and answers."""),

("code", """weather_agent = build_agent(tools=[get_weather], max_turns=6)
run_agent(
    weather_agent,
    "What's the weather in Kathmandu, and should I bring a jacket?",
)"""),

("markdown", """## 3. Why loop instead of one shot?

Without tools, the model guesses. With a loop, it can fetch an observation before advising. Re-run the same question mentally: a one-shot model might invent 22°C and sunshine; the agent above is grounded in `get_weather`."""),

("code", """# Optional contrast: plain ChatGroq with no tools (guessing allowed)
llm = make_llm()
if llm is None:
    print("Skipping — no key.")
else:
    guess = llm.invoke("What's the weather in Kathmandu right now? One sentence.")
    print("ONE-SHOT (no tools):", guess.content)"""),

("markdown", """## 4. Cap the loop

`recursion_limit` stops runaway Plan→Act→Observe cycles. Start strict in class (6–8)."""),

("code", """safe_agent = build_agent(tools=[get_weather], max_turns=4)
run_agent(safe_agent, "Weather in Pokhara — jacket or not?")"""),

("markdown", """## Closing

LangChain handled tool routing; Groq powered the decisions. Class 3 adds chat memory and a multi-tool agent you can extend.

**Next:** Class 3 — Building a Simple Agent."""),

("markdown", """## Challenges

Implement each tool / safeguard with LangChain `@tool` and rebuild the agent so your new tools are included."""),

("markdown", """### Challenge 01 — `convert_currency`
Add `@tool def convert_currency(amount: float, from_currency: str, to_currency: str) -> str` using a small hard-coded rate table. Wire it into `build_agent` and ask a conversion question."""),

("code", """# TODO: define convert_currency, rebuild agent with [get_weather, convert_currency], run_agent(...)
pass"""),

("markdown", """### Challenge 02 — `get_time`
Write `@tool def get_time(timezone: str) -> str` (fake is fine: return a fixed ISO-like string per timezone). Wire it in and test."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 03 — Validate arguments
Inside a tool, reject bad input (e.g. negative `amount`) by returning an error string the agent can observe — do not raise uncaught exceptions."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 04 — `max_turns`
Rebuild with `max_turns=3` (recursion_limit) and confirm the agent still answers a simple weather question."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 05 (stretch) — Two tools in one question
Ask one question that should call both weather and currency (or time) tools before the final answer. Print the full message trace."""),

("code", """# TODO
pass"""),
]

nb.cells = [
    nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
    for kind, src in CELLS
]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
path = "/Users/praveshchapagain/Desktop/genai-foundations-b2/week-5/class-2.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print("Wrote", path)
```

- [ ] **Step 2: Generate the notebook**

Run: `python /tmp/build_week5_class2.py`  
Expected: `Wrote .../week-5/class-2.ipynb`

- [ ] **Step 3: Structural smoke-check**

```bash
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("week-5/class-2.ipynb").read_text())
srcs = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])
for needle in ["create_agent", "ChatGroq", "get_weather", "recursion_limit", "LlamaIndex", "convert_currency"]:
    assert needle in srcs, needle
print("class-2 structural OK:", len(nb["cells"]), "cells")
PY
```

Expected: `class-2 structural OK`

- [ ] **Step 4: Optional live smoke (only if `GROQ_API_KEY` is set)**

```bash
python - <<'PY'
import os
if not os.environ.get("GROQ_API_KEY"):
    print("SKIP live smoke — no GROQ_API_KEY")
else:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def get_weather(location: str) -> str:
        \"\"\"Weather lookup.\"\"\"
        return "temp_c=14, condition=rain"

    agent = create_agent(
        model=ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"]),
        tools=[get_weather],
        system_prompt="Use tools when helpful. Be brief.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Weather in Kathmandu — jacket?"}]},
        config={"recursion_limit": 6},
    )
    print(result["messages"][-1].content)
PY
```

Expected: either `SKIP live smoke` or a short jacket-related answer without traceback.

- [ ] **Step 5: Commit**

```bash
git add week-5/class-2.ipynb
git commit -m "$(cat <<'EOF'
Add Week 5 Class 2 LangChain agent-loop notebook.

Teaches Plan-Act-Observe with ChatGroq tools, recursion limits, and tool challenges.
EOF
)"
```

---

### Task 3: `week-5/class-3.ipynb` — Building a Simple Agent

**Files:**
- Create: `week-5/class-3.ipynb`

**Interfaces:**
- Produces: `calculator(expression: str) -> str`, `mock_search(query: str) -> str` tools; `build_memory_agent(tools)` using `InMemorySaver`; `chat(agent, text, thread_id)` for multi-turn.

- [ ] **Step 1: Write the notebook generator**

Save to `/tmp/build_week5_class3.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
CELLS = [
("markdown", """# Class 3 - Building a Simple Agent

**Week 5: Introduction to AI Agents**

### Learning objectives
By the end of this notebook you will be able to:
- Explain why short-term memory matters across turns.
- Build a multi-tool LangChain agent on Groq (calculator + mock search).
- Add guardrails: recursion limits and safe tool error strings.
- Know when to look at **LangGraph** (complex graphs) or **LlamaIndex** (another agent stack) after this course pattern.

Week 6 will cover retrieval / vector memory — keep memory here to chat history + optional scratch notes."""),

("markdown", """## Setup"""),

("code", """!pip install -q langchain langchain-groq langgraph"""),

("code", """import os
import re
import uuid

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    from google.colab import userdata
    GROQ_API_KEY = GROQ_API_KEY or userdata.get("GROQ_API_KEY")
except Exception:
    pass

if not GROQ_API_KEY:
    print("No GROQ_API_KEY found. Live agent cells will skip.")
else:
    print("Found GROQ_API_KEY. Ready to build the agent.")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


def make_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)


def build_memory_agent(tools, system_prompt=None, max_turns=8):
    llm = make_llm()
    if llm is None:
        return None
    prompt = system_prompt or (
        "You are a helpful multi-tool assistant. "
        "Use calculator for math and mock_search for facts in the catalog. "
        "If a tool errors, explain the problem and ask for a corrected input."
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        checkpointer=InMemorySaver(),
    )
    agent._max_turns = max_turns
    return agent


def chat(agent, text: str, thread_id: str = "week5-class3"):
    if agent is None:
        print("Skipping — no agent / key.")
        return None
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": getattr(agent, "_max_turns", 8),
        },
    )
    final = result["messages"][-1]
    content = getattr(final, "content", final)
    print(content)
    return result"""),

("markdown", """## 1. Why memory matters

Same follow-up question: without a shared `thread_id` + checkpointer, the agent forgets the earlier turn."""),

("code", """@tool
def calculator(expression: str) -> str:
    \"\"\"Evaluate a basic arithmetic expression like '12 * 1.8 + 5'.\"\"\"
    expr = expression.strip()
    if not re.fullmatch(r"[0-9+\\-*/().\\s]+", expr):
        return "Calculator error: only digits and + - * / ( ) are allowed."
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Calculator error: {e}"
    return str(value)


@tool
def mock_search(query: str) -> str:
    \"\"\"Look up a short fact from a tiny local catalog (not the real web).\"\"\"
    catalog = {
        "kathmandu elevation": "Kathmandu sits about 1,400 meters above sea level.",
        "nepal capital": "Kathmandu is the capital of Nepal.",
        "water boil celsius": "Pure water boils at 100°C at standard pressure.",
    }
    q = query.strip().lower()
    for key, value in catalog.items():
        if key in q or q in key:
            return value
    return "No catalog hit. Try queries like 'Nepal capital' or 'Kathmandu elevation'."


demo_tools = [calculator, mock_search]
agent = build_memory_agent(demo_tools)

thread = f"memory-demo-{uuid.uuid4()}"
chat(agent, "Remember that my destination city is Kathmandu.", thread_id=thread)
chat(agent, "Using mock_search, what is the elevation there?", thread_id=thread)"""),

("markdown", """## 2. Two kinds of memory (foundations level)

- **Short-term:** the message list for a `thread_id` (what `InMemorySaver` keeps for this process).
- **Scratch notes:** a string or dict *you* maintain and inject into the next user message when needed.

Vector databases / RAG document memory arrive in Week 6 — do not add them here."""),

("code", """NOTES = {"destination": None}

def note_aware_chat(agent, text: str, thread_id: str = "notes-demo"):
    # Lightweight scratchpad: prepend known notes so the model can use them
    preface = ""
    if NOTES.get("destination"):
        preface = f"(Instructor notes: destination={NOTES['destination']})\\n"
    return chat(agent, preface + text, thread_id=thread_id)

NOTES["destination"] = "Pokhara"
note_aware_chat(agent, "Should I search for the capital or stay focused on my destination?")"""),

("markdown", """## 3. Guardrails

- Cap iterations with `recursion_limit` / `max_turns`.
- Return tool errors as strings (calculator already does).
- Keep the tool list explicit — only register tools you intend to allow."""),

("code", """strict_agent = build_memory_agent(demo_tools, max_turns=5)
chat(strict_agent, "What is 17 * 19? Then remind me of Nepal's capital.", thread_id="guard-demo")"""),

("markdown", """## Closing

You shipped a small agent: tools, memory, and limits. For branching multi-actor workflows, explore **LangGraph** next. If your team standardizes on **LlamaIndex**, the same ideas transfer — tools, a loop, memory, and stop conditions.

**Next week:** Foundations of RAG & Chatbots."""),

("markdown", """## Challenges"""),

("markdown", """### Challenge 01 — `unit_convert`
Add `@tool def unit_convert(value: float, from_unit: str, to_unit: str) -> str` supporting at least `celsius↔fahrenheit` and `km↔miles`. Register it and test."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 02 — Expand `mock_search`
Add three new catalog entries and demonstrate a successful lookup for one of them."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 03 — Harden the calculator
Extend validation (e.g. reject `**`, `//`, or empty input) and show the agent recovering from a bad expression via the error string."""),

("code", """# TODO
pass"""),

("markdown", """### Challenge 04 (optional) — Persistent notes
Update `NOTES` from user text (e.g. parse "my destination is X") and show a later answer that depends on the note."""),

("code", """# TODO
pass"""),
]

nb.cells = [
    nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
    for kind, src in CELLS
]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
path = "/Users/praveshchapagain/Desktop/genai-foundations-b2/week-5/class-3.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print("Wrote", path)
```

- [ ] **Step 2: Generate the notebook**

Run: `python /tmp/build_week5_class3.py`

- [ ] **Step 3: Structural smoke-check**

```bash
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("week-5/class-3.ipynb").read_text())
srcs = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])
for needle in ["InMemorySaver", "calculator", "mock_search", "unit_convert", "LangGraph", "LlamaIndex", "thread_id"]:
    assert needle in srcs, needle
print("class-3 structural OK:", len(nb["cells"]), "cells")
PY
```

- [ ] **Step 4: Commit**

```bash
git add week-5/class-3.ipynb
git commit -m "$(cat <<'EOF'
Add Week 5 Class 3 multi-tool agent notebook.

Covers LangChain memory, calculator/search tools, guardrails, and extension challenges.
EOF
)"
```

---

### Task 4: Wire Notebook + Colab links in `week-5/index.html`

**Files:**
- Modify: `week-5/index.html`

**Interfaces:**
- Consumes: notebooks from Tasks 1–3
- Produces: per-class Deck / Notebook / Colab actions (Blueprint-styled)

- [ ] **Step 1: Add CSS for action links**

Inside the existing `<style>` block (after `.class-card .go` rules), append:

```css
.class-actions{ display:flex; flex-wrap:wrap; gap:10px; margin-top:auto; padding-top:10px; border-top:1px dashed rgba(127,209,255,.25); }
.class-actions a{
  font-family:var(--font-mono); font-size:12px; letter-spacing:.06em; text-transform:uppercase;
  text-decoration:none; color:var(--cyan); border:1px solid rgba(127,209,255,.4);
  padding:7px 12px; background:rgba(5,26,44,.5);
}
.class-actions a:hover{ border-color:var(--orange); color:var(--orange); }
.class-actions a.colab{ color:var(--bg); background:var(--orange); border-color:var(--orange); }
.class-actions a.colab:hover{ background:#ff9d5c; color:var(--bg); }
.class-card .go{ display:none; }
```

- [ ] **Step 2: Convert class cards from single anchors to divs with actions**

Replace the three `<a class="class-card" href="...">...</a>` cards with this pattern (repeat for classes 1–3):

```html
    <div class="class-card">
      <span class="cnum">Class 01</span>
      <h3>What Makes an Agent an Agent?</h3>
      <p>Agents vs. single prompts, autonomy as a spectrum, and a first look at the decision-making loop.</p>
      <div class="class-actions">
        <a href="class-1.html">Open Deck →</a>
        <a href="class-1.ipynb">Notebook</a>
        <a class="colab" href="https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-5/class-1.ipynb" target="_blank" rel="noopener">▶ Open in Colab</a>
      </div>
    </div>
```

Update Class 02 / 03 titles, blurbs, and `class-N` paths the same way. Also lightly update Class 02 card blurb from "function-calling demo" to "LangChain tool-calling demo" if that sentence remains.

- [ ] **Step 3: Add a notebook note under the class grid**

Insert before `.titleblock-note`:

```html
  <div class="titleblock-note" style="margin-bottom:24px;">
    Each class pairs a slide deck (<b>class-N.html</b>) with a runnable notebook (<b>class-N.ipynb</b>).
    Class 1 uses Groq only; Classes 2–3 use LangChain + Groq. Add a Colab secret named <b>GROQ_API_KEY</b> before running live cells.
  </div>
```

- [ ] **Step 4: Verify link paths**

```bash
python - <<'PY'
from pathlib import Path
html = Path("week-5/index.html").read_text()
for n in (1, 2, 3):
    assert f"week-5/class-{n}.ipynb" in html
    assert f'href="class-{n}.html"' in html
    assert f'href="class-{n}.ipynb"' in html
print("index Colab + notebook links OK")
PY
```

- [ ] **Step 5: Commit**

```bash
git add week-5/index.html
git commit -m "$(cat <<'EOF'
Wire Week 5 index to notebooks and Colab.

Adds Blueprint-styled deck/notebook/Colab actions for all three classes.
EOF
)"
```

---

### Task 5: Targeted slide patches (Class 2 + Class 3)

**Files:**
- Modify: `week-5/class-2.html` (demo title/body + challenges framing only)
- Modify: `week-5/class-3.html` (framework guidance paragraph + optional card tag line)

**Interfaces:**
- Must keep slide counts unchanged (`class-2`: 13 slides, `class-3`: 12 slides).
- Edit text inside existing elements only — no new sections.

- [ ] **Step 1: Patch Class 2 demo slide**

Find the Demo Walkthrough slide (`Live Function-Calling Demo`) and update copy to:

- Title: `Live LangChain Tool-Calling <span class="accent">Demo</span>`
- Intro paragraph: `In the notebook, LangChain + Groq turn one user question into a Plan → Act → Observe trace:`
- Keep the four-step weather/jacket example (it still matches `get_weather`).

Also update the Class 02 card description on the index if not done in Task 4.

- [ ] **Step 2: Patch Class 2 challenges intro if needed**

Keep the five challenge bullets as-is (they already match the notebook). If any bullet says "raw function calling," rewrite to LangChain `@tool` wording. Current bullets are tool-name based and can stay.

Optional one-line add under the title (only if it fits without overflow):  
`Implement these in the Class 2 notebook with LangChain tools.`

- [ ] **Step 3: Patch Class 3 framework guidance**

Replace:

```html
<p style="font-size:22px; opacity:.8; margin-top:26px;" data-editable>Guidance: start custom to understand the loop — reach for a framework when you outgrow it.</p>
```

with:

```html
<p style="font-size:22px; opacity:.8; margin-top:26px;" data-editable>Guidance: Class 1 showed the loop by hand; from Class 2 we ship with LangChain + Groq. LangGraph fits complex graphs; LlamaIndex is another full agent stack with the same ideas.</p>
```

Optionally update the Framework card tag from `e.g. LangChain` to `LangChain (this course)` — text only.

- [ ] **Step 4: Count slides still match**

```bash
python - <<'PY'
from pathlib import Path
for name, expected in [("class-2.html", 13), ("class-3.html", 12)]:
    html = Path("week-5", name).read_text()
    count = html.count('<section class="slide"')
    assert count == expected, (name, count, expected)
print("slide counts unchanged")
PY
```

- [ ] **Step 5: Commit**

```bash
git add week-5/class-2.html week-5/class-3.html
git commit -m "$(cat <<'EOF'
Align Week 5 Class 2–3 slides with LangChain labs.

Updates demo wording and framework guidance without changing slide structure.
EOF
)"
```

---

### Task 6: End-to-end verification

**Files:**
- Test only (read): `week-5/class-1.ipynb`, `class-2.ipynb`, `class-3.ipynb`, `index.html`, `class-2.html`, `class-3.html`

- [ ] **Step 1: Run the combined checklist**

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("week-5")
for n in (1, 2, 3):
    assert (root / f"class-{n}.ipynb").exists()
    assert (root / f"class-{n}.html").exists()

index = (root / "index.html").read_text()
for n in (1, 2, 3):
    assert f"week-5/class-{n}.ipynb" in index

c1 = json.loads((root / "class-1.ipynb").read_text())
c1src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in c1["cells"])
assert "langchain" not in c1src.lower()

c2 = "\n".join(
    "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for c in json.loads((root / "class-2.ipynb").read_text())["cells"]
)
assert "create_agent" in c2 and "ChatGroq" in c2

c3 = "\n".join(
    "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for c in json.loads((root / "class-3.ipynb").read_text())["cells"]
)
assert "InMemorySaver" in c3 and "mock_search" in c3

html3 = (root / "class-3.html").read_text()
assert "LangGraph" in html3 and "LlamaIndex" in html3
assert "reach for a framework when you outgrow it" not in html3

print("Week 5 verification passed")
PY
```

Expected: `Week 5 verification passed`

- [ ] **Step 2: Manual browser glance (instructor)**

Open `week-5/index.html`, `class-2.html`, and `class-3.html` in a browser; confirm no obvious overflow on the patched slides and that Colab buttons are visible.

- [ ] **Step 3: Final status commit only if verification found fixes**

If Step 1 required fixes, commit those fixes with a message like `fix: Week 5 verification gaps`. Otherwise stop — no empty commit.

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| `class-1.ipynb` conceptual + Groq, no LangChain | Task 1 |
| `class-2.ipynb` LangChain tools + Groq loop | Task 2 |
| `class-3.ipynb` memory + multi-tool + guardrails | Task 3 |
| LlamaIndex mentioned, not coded | Tasks 2–3 markdown; Task 5 slide |
| Index Notebook + Colab links | Task 4 |
| Class 2–3 slide alignment | Task 5 |
| Missing-key friendly behavior | Tasks 1–3 setup helpers |
| Challenge lists aligned | Tasks 1–3 challenge sections |
| Verification | Task 6 |
