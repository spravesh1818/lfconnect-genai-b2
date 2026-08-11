# Week 5 Class 4 — Long-Term Memory & LangGraph Agents — Design

## Context

Week 5 currently has three classes (agent basics → LangChain tool loop → simple multi-tool agent with short-term memory). Class 3 already names long-term memory and LangGraph as next steps, then jumps to Week 6 RAG. Students need a dedicated Class 4 that makes those ideas concrete without stealing Week 6’s vector/RAG curriculum.

Decisions locked with the instructor:

- Ship **deck + notebook** (same pairing as Classes 1–3).
- Long-term memory = **JSON (or SQLite) note store across sessions** — no embeddings / vector DBs.
- Notebook arc: **Part A linear LangGraph** first, then **Part B multi-agent game reporter** (Researcher → Writer).
- Approach 1: single `class-4.html` + single `class-4.ipynb` with clear Part A / Part B labels.
- Game research uses a **mock catalog**, not live web scraping.

## Goals

- `week-5/class-4.html` Blueprint-themed deck (~13–15 slides) teaching long-term vs short-term memory and LangGraph (state, nodes, edges, checkpoints), then the game-reporter multi-agent idea.
- `week-5/class-4.ipynb` runnable with `GROQ_API_KEY`: Part A linear `remember → recall → answer` graph with JSON persistence + checkpointer; Part B Researcher → Writer game blog pipeline.
- Update `week-5/index.html` with Class 04 + Colab link; refresh hero/sheet copy for four classes.
- Retarget Class 3 “Next” from Week 6 RAG to Class 4; Class 4 “Next” points to Week 6 RAG.

## Non-goals

- No vector databases, embeddings, or RAG pipelines (Week 6).
- No live web scraping / search APIs for the game reporter.
- No separate `class-4a` / `class-4b` notebooks.
- No full LangGraph production ops (deploy, Human-in-the-loop deep dive, streaming UIs).

## Week map (after Class 4)

| Class | Focus | Stack |
|---|---|---|
| 1 | What is an agent | Groq only |
| 2 | Plan → Act → Observe tools | LangChain + Groq |
| 3 | Multi-tool + short-term memory | LangChain + Groq |
| 4 | Long-term notes + LangGraph (linear → multi-agent) | LangGraph + Groq |
| → Week 6 | RAG & chatbots | embeddings / vector search |

## Deck design (`class-4.html`)

Match existing Week 5 Blueprint theme (grid background, Big Shoulders / IBM Plex, cyan/orange, 16:9 fixed stage, deck controls). Reuse Class 1–2 structure as the template (cleaner than the expanded Class 3 file).

Suggested slide arc (~14 sheets):

1. Title — Long-Term Memory & LangGraph Agents  
2. Agenda (Part A memory graph / Part B game reporter)  
3. Recap: Class 3 short-term message memory  
4. Long-term memory without vectors (JSON notes across sessions)  
5. Why graphs: explicit state + nodes beat one opaque loop  
6. LangGraph building blocks — state, nodes, edges, checkpoint  
7. Part A schematic — `remember → recall → answer`  
8. Part B — multi-agent as specialized nodes  
9. Game reporter flow — Researcher → Writer (+ optional Editor in challenges)  
10. Pitfalls — stale notes, invented research, unbounded graphs  
11. Notebook challenges preview  
12. Key takeaways  
13. Next — Week 6 RAG (vectors for document memory)  

Navigation: prev → `class-3.html`; next can link to `../week-6/index.html` or `../week-6/class-1.html`.

## Notebook design (`class-4.ipynb`)

Conventions match Week 5 Classes 2–3: learning objectives, `pip install`, Groq key via env + Colab `userdata`, graceful skip without key, `# TODO` challenges without full solutions. Default model: `llama-3.3-70b-versatile`.

### Part A — Linear long-term memory graph

1. Setup: `langgraph`, `langchain-groq` (and minimal deps).  
2. JSON note store helpers: `load_notes(path)` / `save_note(path, key, value)` writing under a notebook-local path (e.g. `agent_notes.json`).  
3. Graph state (TypedDict): user message, recalled notes, assistant reply.  
4. Nodes:  
   - `remember` — if the user states a durable fact, write it to JSON  
   - `recall` — load notes into state  
   - `answer` — Groq reply grounded in recalled notes  
5. Wire linear edges; compile with `MemorySaver` (or equivalent checkpointer).  
6. Demo: turn 1 saves a preference; new invoke/thread still recalls from JSON file.

### Part B — Game reporter multi-agent

1. Mock catalog of 5–8 games: name, genre, year, 3–5 bullet facts.  
2. `researcher` node: look up game by name (fuzzy/normalized); return structured research string or a clear miss message.  
3. `writer` node: Groq drafts a short blog post **using only** research notes (system prompt forbids inventing facts).  
4. Linear multi-agent graph: `researcher → writer`.  
5. Demo: student input like `"Write a post about Celeste"` → print research + blog draft.

### Challenges

1. Add a new game entry to the catalog and run the reporter on it.  
2. Add an `editor` node after Writer that shortens/tightens tone.  
3. Persist the student’s “favorite game” into the Part A JSON store and have a later answer mention it.  
4. (Stretch) Conditional edge: if research misses, route to a “clarify game name” reply instead of Writer.

## Index + Class 3 wiring

### `week-5/index.html`

- Add Class 04 card (deck + notebook + Colab) pointing at  
  `https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-5/class-4.ipynb`
- Hero: four classes; update titleblock sheet count / copy.
- Notebook note: Class 4 uses LangGraph + Groq; still `GROQ_API_KEY`.

### `week-5/class-3.html`

- Change final “Next Week / RAG” teaser to **Next Class — Long-Term Memory & LangGraph Agents** linking to `class-4.html`.
- Keep Week 6 as Class 4’s forward pointer instead.

## Stack & error handling

- Provider: Groq via `ChatGroq` / LangGraph nodes.  
- Persistence: JSON file only (SQLite optional mention on slides, not required in lab).  
- Missing API key: print guidance; skip live LLM nodes.  
- Missing game: researcher returns an observation string; writer must not hallucinate a full fake catalog entry (prompt + show miss path).  
- Graph recursion / step caps where the API allows.

## Success criteria

- Student can explain short-term vs long-term memory and name JSON notes as one non-vector approach.  
- Student can describe LangGraph state / node / edge / checkpoint at a foundations level.  
- Student can run Part A and see a fact survive across invokes via the JSON file.  
- Student can run Part B and get a blog draft grounded in mock research for a chosen game.  
- Index lists four classes; Class 3 → Class 4 → Week 6 navigation is coherent.

## Testing / verification

- Structural: notebooks mention `StateGraph` (or current LangGraph graph API), JSON helpers, researcher/writer, challenges.  
- No `langchain` import requirement beyond what’s needed for ChatGroq / messages.  
- With `GROQ_API_KEY`, Part A save/recall and Part B reporter complete without uncaught crashes.  
- Index Colab URL for `class-4.ipynb` is correct.  
- Open `class-4.html` on 16:9 stage; no obvious overflow; slide count stable after edits.
