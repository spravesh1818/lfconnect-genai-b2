# Week 5 Class 4 — Long-Term Memory & LangGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Week 5 Class 4 deck + notebook for JSON long-term memory and LangGraph (linear memory graph → multi-agent game reporter), wire the index, and retarget Class 3’s next slide to Class 4.

**Architecture:** Clone the Class 2 Blueprint deck shell for `class-4.html` (14 slides). Build `class-4.ipynb` via nbformat with Part A (`remember → recall → answer` + JSON notes + `InMemorySaver`) and Part B (`researcher → writer` over a mock game catalog). Update `index.html` and Class 3 navigation only.

**Tech Stack:** HTML/CSS/JS (existing Week 5 Blueprint), Python/`nbformat`, `langgraph`, `langchain-groq`, Groq `llama-3.3-70b-versatile`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-week-5-class-4-langgraph-memory-design.md`
- No vector DBs / RAG / live web scrape
- JSON file long-term notes (`agent_notes.json`); graph checkpointer = `InMemorySaver` (same family as Class 3)
- LangGraph API: `StateGraph`, `START`, `END`, `add_node`, `add_edge`, `compile(checkpointer=...)`
- Groq key: env then Colab `userdata`; skip LLM nodes gracefully without key
- Colab: `https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-5/class-4.ipynb`
- Deck template: copy structure/CSS/JS from `week-5/class-2.html` (not the oversized class-3.html)
- Challenge cells: markdown + `# TODO` / `pass` — no full solutions

## File map

| File | Action |
|---|---|
| `week-5/class-4.html` | Create — 14-slide Blueprint deck |
| `week-5/class-4.ipynb` | Create — Part A + Part B notebook |
| `week-5/index.html` | Modify — Class 04 card, four-class copy |
| `week-5/class-3.html` | Modify — Next → Class 4; nav link |

---

### Task 1: `week-5/class-4.ipynb`

**Files:**
- Create: `week-5/class-4.ipynb`

**Interfaces:**
- Produces: `NOTES_PATH`, `load_notes()`, `save_note(key, value)`, `build_memory_graph()`, `build_reporter_graph()`, `GAME_CATALOG`, `lookup_game(name)`

- [ ] **Step 1: Generate notebook** with sections:
  - Title + objectives
  - Setup (`pip install langgraph langchain-groq`, key resolution, `make_llm()`)
  - **Part A:** JSON helpers; `MemoryState` TypedDict (`user_message`, `notes`, `reply`); nodes `remember` / `recall` / `answer`; linear graph; demo save preference then recall on new thread_id (JSON still has the fact)
  - **Part B:** `GAME_CATALOG` (Celeste, Hades, Stardew Valley, Among Us, Minecraft, Zelda BOTW, Valorant, Balatro — name/genre/year/facts); `ReporterState`; `researcher` → `writer`; demo “Write a post about Celeste”; miss-path demo for unknown game
  - Challenges 01–04 per spec
- [ ] **Step 2: Structural smoke-check** — assert `StateGraph`, `remember`, `researcher`, `writer`, `GAME_CATALOG`, `Challenges`, no vector/chroma/pinecone
- [ ] **Step 3: Commit** (unless user said no-commit)

---

### Task 2: `week-5/class-4.html`

**Files:**
- Create: `week-5/class-4.html` (base CSS/JS from `class-2.html`)

**Slide list (14):** Title; Agenda; Class 3 recap; Long-term without vectors; Why graphs; LangGraph building blocks; Part A schematic; Part B multi-agent; Game reporter flow; Pitfalls; Notebook challenges; Takeaways; Next Week 6.

- [ ] **Step 1: Create deck** with DWG `WK5-C4`, nav prev `class-3.html`, next `../week-6/index.html`
- [ ] **Step 2: Verify** `document.querySelectorAll('.slide').length === 14` via a one-liner in browser or count `<section class="slide"`
- [ ] **Step 3: Commit**

---

### Task 3: Index + Class 3 wiring

**Files:**
- Modify: `week-5/index.html`
- Modify: `week-5/class-3.html` (Next slide + bottom nav only)

- [ ] **Step 1: Index** — hero “four classes”; Class 04 card with deck/notebook/Colab; notebook note mentions Class 4 LangGraph; titleblock `4 SHEETS`
- [ ] **Step 2: Class 3** — change Next Week slide to Next Class / Class 4 LangGraph; `nav-next` href → `class-4.html`
- [ ] **Step 3: Verify** Colab URL contains `week-5/class-4.ipynb`; class-3 no longer primary-links Week 6 as next class
- [ ] **Step 4: Commit**

---

### Task 4: End-to-end verification

- [ ] **Step 1: Run checklist**

```python
from pathlib import Path
import json
root = Path("week-5")
assert (root / "class-4.html").exists() and (root / "class-4.ipynb").exists()
html = (root / "class-4.html").read_text()
assert html.count('<section class="slide"') == 14
idx = (root / "index.html").read_text()
assert "week-5/class-4.ipynb" in idx and "Class 04" in idx
c3 = (root / "class-3.html").read_text()
assert "class-4.html" in c3
nb = json.loads((root / "class-4.ipynb").read_text())
src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])
for needle in ["StateGraph", "remember", "researcher", "writer", "GAME_CATALOG", "agent_notes.json"]:
    assert needle in src, needle
assert "chroma" not in src.lower() and "pinecone" not in src.lower()
print("Week 5 Class 4 verification passed")
```

- [ ] **Step 2: Fix any failures; commit if needed**

---

## Spec coverage

| Spec item | Task |
|---|---|
| class-4.ipynb Part A + B | 1 |
| class-4.html deck | 2 |
| index + class-3 next | 3 |
| verification | 4 |
