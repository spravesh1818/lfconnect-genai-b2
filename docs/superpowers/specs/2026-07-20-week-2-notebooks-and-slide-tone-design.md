# Week 2 Notebooks + Slide Tone Pass — Design

## Context

Week 2 ("Introduction to LLMs") currently has three slide decks (`class-1.html`, `class-2.html`, `class-3.html`) and an `index.html`, but no companion notebooks. Week 1 established the pattern this course follows: each class gets a `.ipynb` with learning objectives, a Groq API setup cell, numbered concept sections mixing markdown + runnable code, a closing transition blurb, and a "Challenges" section with 4-5 TODO cells. `week-1/index.html` links each notebook via a local link + a Google Colab deep link into the `spravesh1818/lfconnect-genai-b2` GitHub repo.

The user wants:
1. Week-2 notebooks that give students a hands-on path through the history of language understanding (pre-LLM NLP through transformers) plus exercises, alongside the mechanics the slides already describe (tokenization, sampling, hallucination).
2. `week-2/index.html` wired up with Colab links per class, matching week-1's pattern.
3. A copy-only tone pass on the three existing slide decks so the language reads more human and less like generic AI-generated copy — no structural or layout changes.

## Goals

- Three notebooks (`week-2/class-1.ipynb`, `class-2.ipynb`, `class-3.ipynb`) that are runnable end-to-end (modulo needing a `GROQ_API_KEY` for live-API cells) and match what each slide deck's "Demo Walkthrough" and "Hands-On Preview" slides already promise.
- `class-1.ipynb` additionally covers the pre-LLM history of language understanding (rule-based → statistical → neural → transformer) with a small runnable demo per era, not just narrative text.
- `week-2/index.html` updated with per-class Colab-open links in the existing dark/neon visual style.
- `class-1.html`, `class-2.html`, `class-3.html` get a text-only rewrite of their `data-editable` copy for a more natural, human voice — same structure, same bullet/card counts, same technical claims.

## Non-goals

- No new slides added to any deck.
- No `week-2/assignment.ipynb` capstone (not requested; easy follow-up later).
- No visual/theme changes to `index.html` or the slide decks beyond adding the Colab link markup.
- No change to the underlying technical content of the slides — tone only.

## Notebook designs

All three notebooks follow week-1's conventions: a title + learning-objectives markdown cell, a setup cell (`pip install` + Groq key resolution via Colab secrets with an env-var fallback and a friendly missing-key message), numbered `## N. Section Title` markdown cells paired with code cells, a short closing markdown blurb, then a `## Challenges` section with numbered challenge cells (markdown prompt + acceptance criteria, followed by a `# TODO` starter code cell, no solutions).

### `class-1.ipynb` — What Is a Large Language Model?

Sections:
1. **A short history of getting machines to understand language** — four sub-sections, each with a small runnable demo:
   - Rule-based era: a tiny hand-written pattern-matching responder (ELIZA-style), a few `if`/`in` checks over keywords.
   - Statistical NLP: a bag-of-words / n-gram frequency demo using `collections.Counter`.
   - Neural era: a toy word-vector demo showing analogy arithmetic (`king - man + woman ≈ queen`) using a handful of hand-set low-dimensional vectors and cosine similarity — no real embedding model needed.
   - Transformers/attention: a small numpy self-attention demo over a toy sentence (e.g., "river bank was flooded" vs. "deposited cash at the bank"), showing attention weights shifting with context — mirrors the slide's disambiguation example.
2. **Next-token prediction, toy version** — the `Counter`-based "predict what follows 'the'" demo already described in the slide's Demo Walkthrough section.
3. Closing blurb connecting the toy demos back to real LLMs (parameters instead of counts, but same underlying question).

Challenges (4, matching the slide's Hands-On Preview): extend the corpus and observe prediction shifts; chain predictions to generate a short phrase; feed an out-of-vocabulary word and observe the failure; write a short paragraph in your own words connecting the toy model to real LLMs.

### `class-2.ipynb` — Tokens, Temperature & Context Windows

Sections:
1. Tokenization with `tiktoken` — count tokens for several example sentences (English + one non-English), matching the slide's code box exactly.
2. Sampling live — one Groq call at `temperature=0.1`, one at `temperature=1.2`, same prompt, printed side by side.
3. Cost estimate — a cell computing approximate cost for a sample request given a published per-token rate.

Challenges (4, matching slide): token-count a paragraph and compare to your own estimate; sweep temperature at 0, 0.7, 1.5 and describe the differences; compute the cost of a 5,000-token request; simulate truncating a long conversation to fit a small pretend context window.

### `class-3.ipynb` — Capabilities, Limitations & Responsible AI

Sections:
1. Trigger a hallucination on purpose — the Marie Curie/1922-lecture false-premise prompt from the slide, sent via Groq.
2. Discussion cell: did the model hedge, invent a quote, or correct the false premise? How would you catch this in practice?

Challenges (4, matching slide): design your own trap/false-premise prompt; generate several short stories and compare recurring assumptions for a bias pattern; find a multi-step arithmetic prompt the model gets wrong; draft three responsible-use rules for a hypothetical team.

## `week-2/index.html` changes

Add a `▶ Open in Colab` link to each of the three class cards (and keep the existing `notebook-note` block, updated if needed), pointing at:
`https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-N.ipynb`

Styled using week-2's existing tokens (`--cyan`, `--magenta`, `--panel2`, `--font-mono`) rather than reusing week-1's terminal-theme `colab-link` styling verbatim — same function, deck-appropriate look.

## Slide tone pass

Applies to all `data-editable` text nodes in `class-1.html`, `class-2.html`, `class-3.html`. Rules:
- Text only — no new/removed slides, bullets, cards, or DOM structure.
- Preserve every technical claim as-is; this is a voice/rhythm edit, not a content edit.
- Break up the repeated "X — Y, not Z" / triplet cadence that recurs across the decks; vary sentence length and openers.
- Keep it a teaching voice (a knowledgeable person explaining this to you), not marketing copy — no invented enthusiasm, no new jargon.
- Re-check each edited slide against Mode C fixed-stage limits (16:9 stage, no overflow) since wording length will shift slightly.

## Testing / verification

- Notebooks: each code cell should run without error given a valid `GROQ_API_KEY` (or degrade gracefully with the "no key" message where the slide's own notebooks already do this in week 1). Toy/offline demos (history section, tokenizer counts) must run with no API key at all.
- `index.html`: verify the three Colab links resolve to the correct GitHub paths (can't verify the notebooks are pushed yet, but the URL shape must be correct).
- Slides: open each edited deck in a browser (or visually inspect) to confirm no text overflow and slide count/structure unchanged from before the edit.
