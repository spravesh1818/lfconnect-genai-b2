# Week 5 Agents — Notebooks + Light Slide Alignment — Design

## Context

Week 5 ("Introduction to AI Agents") already has a complete Blueprint-themed set of materials: `index.html` plus three slide decks (`class-1.html`, `class-2.html`, `class-3.html`). The decks promise notebook challenges and demos, but no `.ipynb` files exist yet. Week 4 established the current pairing pattern: each class has a deck, a runnable notebook with Groq via Colab secrets, and index links for local notebook + Open in Colab (`spravesh1818/lfconnect-genai-b2`).

Decisions locked with the instructor:

- Keep the existing three class titles and deck structure (no remap to no-code / agent-patterns as primary titles).
- Approach 1: notebooks-first, with only targeted slide patches where decks contradict the new lab path.
- Introduce LangChain + Groq from Class 2 onward (not raw function-calling as the main student path).
- Prefer LangChain over LangGraph for foundations ease; mention LlamaIndex (and LangGraph) as alternatives students can use later — no LlamaIndex coded labs.

## Goals

- Three notebooks (`week-5/class-1.ipynb`, `class-2.ipynb`, `class-3.ipynb`) runnable end-to-end given a `GROQ_API_KEY`, matching each deck’s Hands-On / Notebook Challenges lists where practical.
- Class 1 stays conceptual with light Groq demos (no LangChain).
- Class 2 teaches Plan → Act → Observe through LangChain tools + a Groq chat model.
- Class 3 builds a small multi-turn agent (tools + short-term memory + iteration guardrails) in LangChain.
- `week-5/index.html` gains Notebook + Colab links in the existing Blueprint visual style (functionally like Week 4).
- Targeted copy patches in Class 2–3 decks so lecture and lab agree on “LangChain from Class 2.”

## Non-goals

- No full deck redesign or theme change.
- No new class titles; no dedicated no-code agent apps class.
- No LlamaIndex or LangGraph deep labs (mention only).
- No Week 5 assignment notebook unless requested later.
- No dual-path “raw Groq loop + LangChain” notebooks (Approach 3 rejected).

## Week map

| Class | Existing deck | New notebook | Stack |
|---|---|---|---|
| 1 | What Makes an Agent an Agent? | `class-1.ipynb` | `groq` only |
| 2 | The Agent Loop — Plan, Act, Observe | `class-2.ipynb` | LangChain + `langchain-groq` (+ LangGraph only if required by the thinnest current agent API) |
| 3 | Building a Simple Agent | `class-3.ipynb` | same as Class 2 |

## Notebook conventions

Follow Week 4 style:

- Title + learning-objectives markdown
- Setup: `pip install` cells, Groq key via Colab `userdata` with env-var fallback, friendly missing-key message (never hardcode keys)
- Numbered concept sections mixing markdown + runnable code
- Closing transition blurb to the next class
- `## Challenges` with numbered prompts + `# TODO` starter cells (no full solutions)
- Tool failures should return an observable string when possible; agent loops must cap iterations

## Notebook designs

### `class-1.ipynb` — What Makes an Agent an Agent?

Learning focus: agent vs single prompt; autonomy as a spectrum; preview of Plan → Act → Observe without a framework.

Sections:

1. Setup (`groq` + key resolution).
2. Single-prompt baseline — one Groq completion for a multi-step question that a single shot handles poorly or incompletely.
3. Manual 2–3 step “pseudo-agent” chain — student/instructor-driven steps that mimic plan then act then observe using plain Python + Groq (no tools API, no LangChain).
4. Autonomy spectrum exercise — classify scenarios (chatbot reply, tool-using assistant, long-running workflow) as low / medium / high autonomy.
5. Closing: what Class 2 will automate with LangChain tools.

Challenges (align to deck):

1. Modify the demo call to ask a different one-sentence question.
2. Write a 3-step manual chain from scratch.
3. Classify five scenarios by autonomy level.
4. (Optional stretch if space) Spot which of two transcripts is “agent-like” and why.

### `class-2.ipynb` — The Agent Loop — Plan, Act, Observe

Learning focus: tool calling as Plan → Act → Observe, implemented with LangChain + Groq.

Sections:

1. Setup: install LangChain + `langchain-groq` (minimal working set); key resolution; short note that LlamaIndex can express the same tool-use pattern.
2. Define tools with `@tool` (start with one calculator-style or lookup tool matching the deck’s live demo spirit).
3. Bind tools to a Groq chat model and run a short agent / tool loop so students see plan (model picks a tool), act (Python runs it), observe (result returns to the model).
4. Why loop beats one-shot — side-by-side or narrative comparison on a question that needs a tool.
5. Safety: `max_turns` / max iterations before continuing.
6. Closing: Class 3 adds memory and a fuller multi-tool agent.

Challenges (align to deck, expressed in LangChain):

1. Add a second tool: `convert_currency(amount, from, to)`.
2. Write a `get_time(timezone)` tool from scratch and wire it in.
3. Validate tool arguments before executing them.
4. Add a `max_turns` safety counter to the loop / agent config.
5. Stretch — one question that triggers two different tools in sequence.

### `class-3.ipynb` — Building a Simple Agent

Learning focus: ship a small agent with tools, short-term memory, and guardrails.

Sections:

1. Setup (reuse Class 2 stack).
2. Why memory matters — same question with/without chat history.
3. Two kinds of memory at foundations level: short-term conversation messages vs simple longer notes / scratchpad (keep scratchpad light; no vector DB — that is Week 6).
4. Anatomy of the agent: system prompt, tools, loop, stop conditions.
5. Build end-to-end: calculator + mock search (+ room for a third tool in challenges).
6. Guardrails: max iterations, tool allowlist / failed-tool messages.
7. Closing: point forward to Week 6 RAG; note LangGraph for complex graphs and LlamaIndex as an alternate stack.

Challenges (align to deck):

1. Add a third tool: `unit_convert(value, from_unit, to_unit)`.
2. Expand mock search with three new lookup entries.
3. Harden the calculator against invalid expressions.
4. (Optional) Persist a short “notes” string across turns and show it affecting a later answer.

## Index + slide patches

### `week-5/index.html`

- Add per-class Notebook + Colab actions (or equivalent Blueprint-styled links) pointing at:
  `https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-5/class-N.ipynb`
- Keep existing theme, diagram, and class titles; only extend cards/meta for lab access (Week 4 functional parity, Week 5 visual language).

### Targeted deck copy (not a redesign)

- **Class 2:** Demo Walkthrough + Notebook Challenges wording should refer to LangChain tools / agent wiring, not raw provider function-calling JSON as the student path.
- **Class 3:** “Framework or Roll Your Own?” guidance flips to: students already understand the loop conceptually from Class 1–2; LangChain is how this course ships agents; LangGraph and LlamaIndex are named alternatives.
- Preserve Blueprint layout, slide counts, and visual system unless a wording change forces a minor overflow fix.

## Success criteria

- A student can explain agent vs single prompt and name Plan → Act → Observe.
- A student can define a LangChain tool, run it with Groq, and add a second tool.
- A student can run a multi-turn agent with chat memory and an iteration cap.
- Decks and notebooks do not contradict each other on when frameworks appear (Class 2+).

## Testing / verification

- Class 1 offline-ish cells (classification exercise) run without an API key; live Groq cells degrade with the missing-key message.
- Class 2–3: with a valid `GROQ_API_KEY`, tool demos and the multi-turn agent complete without uncaught crashes; intentional bad tool input returns a string observation or a clear error cell, not a hung loop.
- Index Colab URLs use the correct `week-5/class-N.ipynb` paths.
- Open patched decks and confirm no text overflow on 16:9 stage after wording edits.
