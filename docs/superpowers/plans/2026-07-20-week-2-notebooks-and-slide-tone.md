# Week 2 Notebooks + Slide Tone Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three companion notebooks to `week-2/` (history-of-language-understanding demos + the tokenization/temperature/hallucination demos the slides already promise), wire Colab links into `week-2/index.html`, and rewrite the copy inside the three existing slide decks to read more human — no structural changes anywhere.

**Architecture:** Notebooks are built via a short nbformat-based Python generator script (run once, output committed, generator discarded — it's not a repo artifact). Slide edits are plain text substitutions inside existing `data-editable` elements, verified by a structure-only diff so no tag/attribute/slide-count drifts in. `index.html` gets its class cards converted from whole-card anchors to divs with an action-link row (matching the pattern `week-1/index.html` already uses), plus a Colab deep link per class.

**Tech Stack:** Python 3 (`nbformat`, `nbclient`, `nbconvert`, `ipykernel`, `tiktoken`, `numpy` — all already available or installed in Task 1), plain HTML/CSS (no build step, matches existing files).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-week-2-notebooks-and-slide-tone-design.md`
- No new slides added to any deck (`class-1.html`: 13 slides, `class-2.html`: 14 slides, `class-3.html`: 13 slides — must stay exactly these counts).
- No `week-2/assignment.ipynb` capstone — out of scope for this plan.
- Notebooks must run cell-by-cell with **no** `GROQ_API_KEY` set and exit with no errors — live-API cells must print a friendly "no key" message instead of raising, exactly like `week-1`'s notebooks already do.
- Colab links point at `https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-N.ipynb` (confirmed GitHub remote: `spravesh1818/lfconnect-genai-b2`).
- Slide edits are text-only inside existing `data-editable` nodes — no new/removed DOM elements, no attribute changes, no theme/color changes.

---

## Task 1: `week-2/class-1.ipynb` — What Is a Large Language Model?

**Files:**
- Create: `week-2/class-1.ipynb`

**Interfaces:**
- Produces: a notebook with a `predict_next(word)` function (Section 2) and an `eliza_reply(text)`, `bag_of_words_score(text, positive_words, negative_words)`, `cosine_sim(a, b)`, `attention_weights(words, vectors)` function each, all self-contained (no cross-notebook imports).

- [ ] **Step 1: Write the notebook generator script**

Save this to a scratch path (e.g. `/tmp/build_class1.py` — this script is a one-time generator, not a repo file):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

CELLS = [
("markdown", """# Class 1 — What Is a Large Language Model?
**Week 2: Introduction to LLMs — "The Black Box / Neural Signal"**

### Learning objectives
By the end of this notebook you will be able to:
- Trace the pre-LLM history of getting machines to "understand" language: rule-based, statistical, neural, transformer
- Build a tiny, from-scratch demo of each era's core idea and see why each one gave way to the next
- Build a toy next-token predictor using word-frequency counts
- Connect the toy demos to what a real LLM does at inference time

Run each cell in order with **Shift+Enter**. No API key needed — every demo in this notebook runs on toy data, locally."""),

("markdown", """## Setup
No external packages beyond `numpy` are required for this class."""),

("code", """import sys
import numpy as np

print(f"Python version: {sys.version.split()[0]}")
print("Setup complete — no API key needed for this notebook.")"""),

("markdown", """## 1. A Short History of Getting Machines to Understand Language
Four eras, one thread: each new approach fixed a real limitation of the one before it. We'll build a tiny, honest demo of each — not a real system, just enough code to feel the underlying idea."""),

("markdown", """### 1.1 Rule-based era: pattern matching, not understanding
The earliest chatbots didn't predict anything — they matched keywords against hand-written rules and picked a canned response. ELIZA (1966) is the classic example: it played a Rogerian therapist by echoing your words back as a question."""),

("code", """import re

RULES = [
    (r"\\bI (feel|am) (.*)", "Why do you {0} {1}?"),
    (r"\\bmy (mother|father|mom|dad)\\b", "Tell me more about your family."),
    (r"\\bI need (.*)", "Why do you need {0}?"),
]

def eliza_reply(text):
    for pattern, template in RULES:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return template.format(*m.groups())
    return "Tell me more about that."

for line in ["I feel tired", "my mother called today", "I need a break", "the weather is nice"]:
    print(f"you:   {line}")
    print(f"eliza: {eliza_reply(line)}\\n")"""),

("markdown", """`eliza_reply` doesn't know what "tired" means — it just rearranges your own words into a question. Say something with none of these keywords ("the weather is nice") and you fall through to the generic default. That fallback line *is* the ceiling of rule-based systems: every input the author didn't anticipate needs its own hand-written rule."""),

("markdown", """### 1.2 Statistical era: learn patterns from a corpus
By the 1990s the field moved from hand-written rules to counting: given a labeled body of text, which words show up in which category? A "bag of words" classifier below scores a message as spam-like or not, purely from word overlap — no rules, no grammar, just counts."""),

("code", """from collections import Counter

spam_words = set("free money now click win prize offer".split())
ham_words = set("meeting schedule report project update team".split())

def bag_of_words_score(text, positive_words, negative_words):
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    return pos - neg

messages = [
    "click now to win free money",
    "let's schedule the project meeting",
    "free prize offer, click to win",
]
for msg in messages:
    score = bag_of_words_score(msg, spam_words, ham_words)
    label = "SPAM" if score > 0 else "not spam"
    print(f"{label:9} (score {score:+d}) — {msg!r}")"""),

("markdown", """This "bag of words" throws away word order entirely — "dog bites man" and "man bites dog" score identically. That's the statistical era's core trade-off: more general than hand-written rules, but it treats language as an unordered pile of evidence, not a structure."""),

("markdown", """### 1.3 Neural era: words become vectors
Word embeddings (word2vec, GloVe, ~2013) represent each word as a point in space, learned so that words used in similar contexts land near each other. The famous party trick: vector arithmetic captures relationships — "king" minus "man" plus "woman" lands near "queen". Below is a hand-built miniature of that idea. Real embeddings have hundreds of dimensions, learned from billions of words; these four are made up by hand, just to show the shape of the idea."""),

("code", """# hand-set toy vectors — real embeddings are learned, not hand-written
vectors = {
    "king":  np.array([0.9, 0.8, 0.1]),
    "man":   np.array([0.8, 0.2, 0.1]),
    "woman": np.array([0.2, 0.2, 0.1]),
    "queen": np.array([0.3, 0.8, 0.1]),
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

result = vectors["king"] - vectors["man"] + vectors["woman"]
print("king - man + woman ≈", result.round(2))
print()
for word, vec in vectors.items():
    print(f"similarity to result: {word:6} -> {cosine_sim(result, vec):.3f}")"""),

("markdown", """"queen" comes out nearest — the toy vectors capture a real relationship (gender, as a direction in space) without ever being told the rule "a queen is a female king." This geometric view of meaning is what later gets fed into transformers."""),

("markdown", """### 1.4 Transformer era: attention over vectors
Word embeddings alone are static — "bank" gets the same vector whether you mean a riverbank or a financial bank. Self-attention lets each word's representation shift based on its neighbors. Below is a tiny numpy version: dot products between word vectors, turned into weights with softmax — the same core computation a transformer's attention layer runs, just with random (not learned) numbers."""),

("code", """words_a = ["deposited", "cash", "at", "the", "bank"]
words_b = ["sat", "by", "the", "river", "bank"]

np.random.seed(0)
base = {w: np.random.randn(4) for w in set(words_a + words_b) - {"bank"}}
base["bank"] = np.random.randn(4)

def attention_weights(words, vectors):
    vecs = np.array([vectors[w] for w in words])
    scores = vecs @ vecs.T
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

for words in (words_a, words_b):
    weights = attention_weights(words, base)
    bank_idx = words.index("bank")
    print(f"sentence: {' '.join(words)}")
    for w, weight in zip(words, weights[bank_idx]):
        print(f"  'bank' attends to {w:10} -> {weight:.2f}")
    print()"""),

("markdown", """The vectors here are random, not learned — so don't read meaning into which word "wins" in this toy. What matters is the mechanism: attention is dot products between vectors, turned into weights by softmax. A real transformer learns vectors where "deposited" and "cash" genuinely pull "bank" toward its financial meaning — this is the same computation, just with meaningful numbers instead of random ones."""),

("markdown", """## 2. Next-Token Prediction, Toy Version
Same idea as Section 1.2's spam counter, aimed at a different question: instead of counting which words label a message as spam, count which word tends to *follow* another. That's already most of what a next-token predictor does."""),

("code", """from collections import defaultdict, Counter

corpus = "the cat sat the cat ran the dog sat"
follows = defaultdict(Counter)
words = corpus.split()
for a, b in zip(words, words[1:]):
    follows[a][b] += 1

for word, counter in follows.items():
    print(f"{word:6} -> {dict(counter)}")

def predict_next(word):
    counter = follows.get(word)
    if not counter:
        return None
    return counter.most_common(1)[0][0]

print("\\nPredict next after 'the':", predict_next("the"))"""),

("markdown", """### Week 2, Class 1 — closed
Four eras, one thread: each new approach fixed a limitation of the last. Rules couldn't scale past what someone anticipated. Counting patterns ignored order and meaning. Static vectors couldn't adapt to context. Attention did. Stack enough attention layers, train on enough text, and you get the model behind the rest of this week's slide deck."""),

("markdown", """## Challenges
Work through these in order. No solutions are provided — each starter cell has a `# TODO` marking where your code goes."""),

("markdown", """### Challenge 1 — Extend the Corpus
Add at least 3 more sentences to the `corpus` string from Section 2 (reuse the same words plus at least one new word), rebuild `follows`, and print the new `follows['the']` counter plus `predict_next('the')`.

**Acceptance criteria:** your corpus has at least 4 sentences total; you print the updated `follows['the']` counter and the new prediction."""),

("code", """# TODO: extend corpus, rebuild follows, print follows['the'] and predict_next('the')"""),

("markdown", """### Challenge 2 — Chain Predictions Into a Phrase
Write a function `generate(start_word, steps)` that calls `predict_next` repeatedly, appending each predicted word, and stops early if a word has no known follower. Run it starting from `'the'` for 5 steps.

**Acceptance criteria:** returns a list of words starting with `start_word`, length at most `steps + 1`, and stops cleanly (no crash) when `predict_next` returns `None`."""),

("code", """# TODO: write generate(start_word, steps) using predict_next, then call generate('the', 5)"""),

("markdown", """### Challenge 3 — Break the Toy Predictor
Call `predict_next` on a word that never appears in your corpus (e.g. `'giraffe'`). Print what happens, then modify `predict_next` so it returns a readable string like `"<no data>"` instead of `None` for unseen words.

**Acceptance criteria:** calling `predict_next` on an out-of-vocabulary word no longer returns a bare `None` — it returns a readable message."""),

("code", """# TODO: call predict_next on an unseen word, then update it to return a readable message instead of None"""),

("markdown", """### Challenge 4 — Connect the Toy to Real LLMs
Write 2-4 sentences (right in this cell, replacing the italic prompt) answering: (a) what `follows` and a real LLM's parameters have in common, and (b) one thing a real LLM can do that this toy model structurally cannot, and why. (Hint: think about corpus size vs. training-set size, and single-word context vs. long context.)

*(Write your answer here.)*"""),
]

for kind, source in CELLS:
    if kind == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(source))
    else:
        nb.cells.append(nbf.v4.new_code_cell(source))

nbf.write(nb, "week-2/class-1.ipynb")
print("wrote week-2/class-1.ipynb with", len(nb.cells), "cells")
```

- [ ] **Step 2: Install notebook tooling and run the generator**

Run:
```bash
pip install -q nbformat nbclient nbconvert ipykernel
python3 /tmp/build_class1.py
```
Expected: `wrote week-2/class-1.ipynb with 27 cells`

- [ ] **Step 3: Execute the notebook end-to-end and verify no errors**

Run:
```bash
jupyter nbconvert --to notebook --execute --output /tmp/class1-executed.ipynb week-2/class-1.ipynb
```
Expected: exits 0, no `Error` printed to terminal (an `[NbConvertApp] Writing ...` success line at the end).

- [ ] **Step 4: Verify key outputs are correct**

Run:
```bash
python3 -c "
import nbformat
nb = nbformat.read('/tmp/class1-executed.ipynb', as_version=4)
text = ''
for cell in nb.cells:
    if cell.cell_type == 'code':
        for out in cell.get('outputs', []):
            text += out.get('text', '')
assert \"similarity to result: queen  -> 1.000\" in text, 'word-vector analogy did not resolve to queen'
assert \"Predict next after 'the': cat\" in text, 'toy next-token predictor did not predict cat'
assert 'eliza: Why do you feel tired?' in text, 'eliza rule did not fire correctly'
print('all key outputs verified')
"
```
Expected: `all key outputs verified`

- [ ] **Step 5: Commit**

```bash
git add week-2/class-1.ipynb
git commit -m "$(cat <<'EOF'
Add week-2 class-1 notebook: history of language understanding

Covers rule-based, statistical, neural, and transformer eras with a
small runnable demo each, then the next-token toy predictor the
class-1 slide deck already references, closing with four challenges.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `week-2/class-2.ipynb` — Tokens, Temperature & Context Windows

**Files:**
- Create: `week-2/class-2.ipynb`

**Interfaces:**
- Produces: `call_llm(prompt, temperature=0.7, model=..., max_tokens=80)`, `estimate_cost(input_tokens, output_tokens)`, module-level `enc` (a `tiktoken` encoding).

- [ ] **Step 1: Write the notebook generator script**

Save to a scratch path (e.g. `/tmp/build_class2.py`):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

CELLS = [
("markdown", """# Class 2 — Tokens, Temperature & Context Windows
**Week 2: Introduction to LLMs — "The Black Box / Neural Signal"**

### Learning objectives
By the end of this notebook you will be able to:
- Count real tokens with `tiktoken` and see why token count isn't word count
- Compare low- vs. high-temperature output from a real LLM call
- Estimate the dollar cost of an API request from its token count
- Simulate trimming a conversation to fit inside a small context window

Section 1 works with no API key. Sections 2-3 call a real model via Groq and need a `GROQ_API_KEY` to produce live output — see Setup."""),

("markdown", """## Setup
**Running in Google Colab:**
1. Get a free key from https://console.groq.com/keys
2. Click the key icon (🔑 Secrets) in the left sidebar
3. Add a secret named `GROQ_API_KEY`, paste your key, toggle **Notebook access** on
4. Run the two setup cells below

**Elsewhere:** set `GROQ_API_KEY` as an environment variable before launching Jupyter."""),

("code", """!pip install -q groq tiktoken"""),

("code", """import os

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print(
        "No API key found — Section 1 still works.\\n"
        "In Colab: add a secret named GROQ_API_KEY via the 🔑 Secrets panel and enable notebook access.\\n"
        "Elsewhere: set GROQ_API_KEY as an environment variable before launching Jupyter."
    )
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    print("GROQ_API_KEY loaded — live sections will work.")"""),

("markdown", """## 1. Counting Real Tokens
Models don't see words — they see tokens, sub-word chunks produced by a tokenizer. `tiktoken` is OpenAI's tokenizer library; Groq's Llama models use a different tokenizer under the hood, but `cl100k_base` is close enough to build intuition, and needs no API call."""),

("code", """import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

examples = [
    "Generative AI is transformative.",
    "supercalifragilisticexpialidocious",
    "La inteligencia artificial generativa es transformadora.",
]

for text in examples:
    tokens = enc.encode(text)
    print(f"{len(tokens):>3} tokens — {text!r}")
    print(f"     pieces: {[enc.decode([t]) for t in tokens]}\\n")"""),

("markdown", """Notice the Spanish sentence uses noticeably more tokens for a similar-length idea — tokenizers are typically trained mostly on English text, so other languages split into smaller, less efficient pieces."""),

("markdown", """## 2. Same Prompt, Two Temperatures (Live)
This cell sends one prompt to Groq twice: once at `temperature=0.1`, once at `temperature=1.2`. Needs `GROQ_API_KEY`."""),

("code", """def call_llm(prompt, temperature=0.7, model="llama-3.3-70b-versatile", max_tokens=80):
    \"\"\"Send a single-turn prompt to Groq at a given temperature and return the text, or an error message.\"\"\"
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq: {e}"

prompt = "Finish this sentence: The old lighthouse keeper opened the door and saw"

if os.environ.get("GROQ_API_KEY"):
    print("temperature = 0.1:\\n", call_llm(prompt, temperature=0.1))
    print("\\ntemperature = 1.2:\\n", call_llm(prompt, temperature=1.2))
else:
    print("Set GROQ_API_KEY, then re-run this cell to see live output at both temperatures.")"""),

("markdown", """## 3. Estimating a Bill
Providers bill per token, usually with separate input/output rates. Given a hypothetical rate, estimate the cost of a request."""),

("code", """# hypothetical pricing for this exercise — check your provider's actual published rates
INPUT_RATE_PER_1K = 0.05   # USD per 1,000 input tokens
OUTPUT_RATE_PER_1K = 0.08  # USD per 1,000 output tokens

def estimate_cost(input_tokens, output_tokens):
    return (input_tokens / 1000 * INPUT_RATE_PER_1K) + (output_tokens / 1000 * OUTPUT_RATE_PER_1K)

sample_input_tokens = len(enc.encode(prompt))
sample_output_tokens = 80  # matches max_tokens above

cost = estimate_cost(sample_input_tokens, sample_output_tokens)
print(f"input tokens: {sample_input_tokens}, output tokens (max): {sample_output_tokens}")
print(f"estimated cost: ${cost:.5f}")"""),

("markdown", """### Week 2, Class 2 — closed
Every call you make to an LLM API is really three numbers underneath: how many tokens went in, how many came out, and how randomly they were sampled. You now know how to measure and control all three."""),

("markdown", """## Challenges
Work through these in order. No solutions are provided — each starter cell has a `# TODO` marking where your code goes. Challenge 2 needs your own `GROQ_API_KEY`."""),

("markdown", """### Challenge 1 — Token-Count a Paragraph
Write your own 2-3 sentence paragraph. Guess how many tokens it will be, then encode it with `enc.encode` and print the actual count next to your guess.

**Acceptance criteria:** prints your guess and the real token count for the same paragraph."""),

("code", """# TODO: write a paragraph, guess its token count, then print your guess vs. enc.encode(paragraph)"""),

("markdown", """### Challenge 2 — Sweep Temperature
Call `call_llm` with the same prompt at `temperature` 0, 0.7, and 1.5. Print all three outputs and, in a comment, describe how they differ.

**Acceptance criteria:** prints three completions at three different temperatures, plus a one-line comment describing the difference."""),

("code", """# TODO: call call_llm(prompt, temperature=...) at 0, 0.7, and 1.5; print all three plus a comment"""),

("markdown", """### Challenge 3 — Estimate a 5,000-Token Request
Using `estimate_cost` from Section 3, compute the cost of a request with 4,000 input tokens and 1,000 output tokens.

**Acceptance criteria:** prints a single dollar-amount cost for 4,000 input / 1,000 output tokens."""),

("code", """# TODO: call estimate_cost(4000, 1000) and print the result"""),

("markdown", """### Challenge 4 — Simulate Context-Window Truncation
Given a list of chat turns (strings) and a small pretend token budget, write `fit_to_budget(turns, budget)` that drops the *oldest* turns (from the front of the list) until the total token count (via `enc.encode`) fits within budget. Test it on the 6 turns below with `budget=50`.

**Acceptance criteria:** returns a list that is a suffix of `turns`, and the total token count of the returned turns is at most `budget`."""),

("code", """turns = [
    "Hi, I need help with my order.",
    "Sure, what's your order number?",
    "It's 48213.",
    "Let me look that up for you.",
    "It shipped yesterday and should arrive Friday.",
    "Great, thank you!",
]

# TODO: write fit_to_budget(turns, budget) using enc.encode to count tokens per turn,
# dropping oldest turns until the total fits, then call fit_to_budget(turns, 50)"""),
]

for kind, source in CELLS:
    if kind == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(source))
    else:
        nb.cells.append(nbf.v4.new_code_cell(source))

nbf.write(nb, "week-2/class-2.ipynb")
print("wrote week-2/class-2.ipynb with", len(nb.cells), "cells")
```

- [ ] **Step 2: Install notebook tooling and run the generator**

Run:
```bash
pip install -q nbformat nbclient nbconvert ipykernel groq tiktoken
python3 /tmp/build_class2.py
```
Expected: `wrote week-2/class-2.ipynb with 21 cells`

- [ ] **Step 3: Execute the notebook with no API key set and verify no errors**

Run:
```bash
unset GROQ_API_KEY
jupyter nbconvert --to notebook --execute --output /tmp/class2-executed.ipynb week-2/class-2.ipynb
```
Expected: exits 0, no unhandled `Error` in terminal output.

- [ ] **Step 4: Verify key outputs are correct**

Run:
```bash
python3 -c "
import nbformat
nb = nbformat.read('/tmp/class2-executed.ipynb', as_version=4)
text = ''
for cell in nb.cells:
    if cell.cell_type == 'code':
        for out in cell.get('outputs', []):
            text += out.get('text', '')
assert 'tokens —' in text, 'tiktoken section did not print token counts'
assert 'Set GROQ_API_KEY, then re-run this cell' in text, 'temperature demo did not degrade gracefully without a key'
assert 'estimated cost: \$' in text, 'cost estimate section did not print a cost'
print('all key outputs verified')
"
```
Expected: `all key outputs verified`

- [ ] **Step 5: Commit**

```bash
git add week-2/class-2.ipynb
git commit -m "$(cat <<'EOF'
Add week-2 class-2 notebook: tokens, temperature, context windows

Matches the class-2 slide deck's demo walkthrough exactly — tiktoken
counting, a live low-vs-high-temperature comparison via Groq, and a
cost estimate — plus four closing challenges.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `week-2/class-3.ipynb` — Capabilities, Limitations & Responsible AI

**Files:**
- Create: `week-2/class-3.ipynb`

**Interfaces:**
- Produces: `call_llm(prompt, system_prompt=None, model=..., max_tokens=200)`.

- [ ] **Step 1: Write the notebook generator script**

Save to a scratch path (e.g. `/tmp/build_class3.py`):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

CELLS = [
("markdown", """# Class 3 — Capabilities, Limitations & Responsible AI
**Week 2: Introduction to LLMs — "The Black Box / Neural Signal"**

### Learning objectives
By the end of this notebook you will be able to:
- Deliberately trigger a hallucination and recognize what it looks like
- Reason about *why* a model fabricates instead of declining
- Practice designing prompts that probe for bias and reasoning failures
- Draft a short responsible-use policy

This notebook needs a `GROQ_API_KEY` for its live cells (see Setup); the discussion and writing cells work regardless."""),

("markdown", """## Setup
**Running in Google Colab:**
1. Get a free key from https://console.groq.com/keys
2. Click the key icon (🔑 Secrets) in the left sidebar
3. Add a secret named `GROQ_API_KEY`, paste your key, toggle **Notebook access** on
4. Run the two setup cells below

**Elsewhere:** set `GROQ_API_KEY` as an environment variable before launching Jupyter."""),

("code", """!pip install -q groq"""),

("code", """import os

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print(
        "No API key found — the discussion and writing cells still work.\\n"
        "In Colab: add a secret named GROQ_API_KEY via the 🔑 Secrets panel and enable notebook access.\\n"
        "Elsewhere: set GROQ_API_KEY as an environment variable before launching Jupyter."
    )
else:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    print("GROQ_API_KEY loaded — live cells will work.")"""),

("markdown", """## 1. Triggering a Hallucination, on Purpose
The prompt below embeds a false premise: Marie Curie died in 1934, decades before language models existed, so there is no 1922 lecture about them. Watch whether the model corrects the premise or plays along."""),

("code", """def call_llm(prompt, system_prompt=None, model="llama-3.3-70b-versatile", max_tokens=200):
    \"\"\"Send a prompt to Groq and return the text, or an error message.\"\"\"
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set."
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq: {e}"

prompt = (
    "What did Marie Curie say about large language models "
    "in her 1922 lecture?"
)

if os.environ.get("GROQ_API_KEY"):
    print(call_llm(prompt))
else:
    print("Set GROQ_API_KEY, then re-run this cell to see the model's actual response.")"""),

("markdown", """## 2. Discussion
Re-run the cell above a few times if you have a key, then answer here (replace this text):
- Did the model correct the false premise (Curie died in 1934), hedge, or invent a plausible-sounding quote?
- If it invented a quote, what made the fabrication *sound* credible?
- How would you catch this in a real application, before a user saw it?

*(Write your answer here.)*"""),

("markdown", """### Week 2, Class 3 — closed
You've now seen the mechanism (Class 1), measured its inputs and outputs (Class 2), and watched it fail on purpose (Class 3). That's the full arc of "what is an LLM, really" — Week 3 moves on to building with one."""),

("markdown", """## Challenges
Work through these in order. No solutions are provided — write your own prompts and observations directly in each cell."""),

("markdown", """### Challenge 1 — Design Your Own Trap Prompt
Write a new false-premise question (not about Marie Curie) designed to probe for hallucination — e.g. attribute a real quote to the wrong person, or ask about an event that didn't happen. Run it through `call_llm` and note what happened.

**Acceptance criteria:** prints your prompt, the model's response, and a one-line note on whether it caught the false premise."""),

("code", """# TODO: write your own false-premise prompt, call call_llm(prompt), and print the result plus your note"""),

("markdown", """### Challenge 2 — Spot a Bias Pattern
Generate 4-5 short stories from the same one-line prompt (e.g. "Write a two-sentence story about a nurse going to work.") and compare them for recurring assumptions (gender, names, setting).

**Acceptance criteria:** prints all 4-5 generated stories together, plus a one-line note on any pattern you noticed."""),

("code", """# TODO: call call_llm on the same one-line prompt 4-5 times, print all outputs, and note any recurring pattern"""),

("markdown", """### Challenge 3 — Break the Model's Math
Find a multi-step arithmetic or logic prompt (e.g. multi-digit multiplication, or a multi-step word problem) that the model gets wrong.

**Acceptance criteria:** prints the prompt, the model's answer, and the correct answer, with the two clearly different."""),

("code", """# TODO: write a multi-step math/logic prompt, call call_llm, and print its answer next to the correct answer"""),

("markdown", """### Challenge 4 — Draft a Responsible-Use Policy
Write three concrete rules for a hypothetical team adopting LLMs at work (e.g. for customer-support drafts). Ground at least one rule in something you saw fail earlier in this notebook.

*(Write your three rules here.)*"""),
]

for kind, source in CELLS:
    if kind == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(source))
    else:
        nb.cells.append(nbf.v4.new_code_cell(source))

nbf.write(nb, "week-2/class-3.ipynb")
print("wrote week-2/class-3.ipynb with", len(nb.cells), "cells")
```

- [ ] **Step 2: Install notebook tooling and run the generator**

Run:
```bash
pip install -q nbformat nbclient nbconvert ipykernel groq
python3 /tmp/build_class3.py
```
Expected: `wrote week-2/class-3.ipynb with 16 cells`

- [ ] **Step 3: Execute the notebook with no API key set and verify no errors**

Run:
```bash
unset GROQ_API_KEY
jupyter nbconvert --to notebook --execute --output /tmp/class3-executed.ipynb week-2/class-3.ipynb
```
Expected: exits 0, no unhandled `Error` in terminal output.

- [ ] **Step 4: Verify key outputs are correct**

Run:
```bash
python3 -c "
import nbformat
nb = nbformat.read('/tmp/class3-executed.ipynb', as_version=4)
text = ''
for cell in nb.cells:
    if cell.cell_type == 'code':
        for out in cell.get('outputs', []):
            text += out.get('text', '')
assert 'Set GROQ_API_KEY, then re-run this cell' in text, 'hallucination demo did not degrade gracefully without a key'
print('all key outputs verified')
"
```
Expected: `all key outputs verified`

- [ ] **Step 5: Commit**

```bash
git add week-2/class-3.ipynb
git commit -m "$(cat <<'EOF'
Add week-2 class-3 notebook: capabilities, limits, responsible AI

Matches the class-3 slide deck's demo walkthrough — the Marie Curie
false-premise hallucination trigger via Groq — plus four closing
challenges on trap prompts, bias, broken math, and use policy.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `week-2/index.html` — Colab links per class

**Files:**
- Modify: `week-2/index.html:96-127` (classes block), `week-2/index.html:60-74` (`.class-card`/`.class-arrow` CSS)

**Interfaces:**
- Consumes: `week-2/class-1.ipynb`, `class-2.ipynb`, `class-3.ipynb` (from Tasks 1-3) via Colab deep links.

- [ ] **Step 1: Update the class-card CSS to support an action row**

In `week-2/index.html`, find:
```css
.class-arrow{font-family:var(--font-mono);font-size:14px;color:var(--cyan);letter-spacing:.08em;text-transform:uppercase;white-space:nowrap;align-self:center;}
```
Replace with:
```css
.class-actions{display:flex;gap:16px;align-items:center;margin-top:20px;flex-wrap:wrap;}
.class-arrow{font-family:var(--font-mono);font-size:14px;color:var(--cyan);letter-spacing:.08em;text-transform:uppercase;white-space:nowrap;text-decoration:none;border:1px solid rgba(34,211,238,.4);padding:9px 16px;border-radius:8px;transition:color .2s,border-color .2s;}
.class-arrow:hover{color:var(--text);border-color:var(--cyan);}
.colab-link{font-family:var(--font-mono);font-size:14px;letter-spacing:.05em;color:var(--bg);background:var(--amber);text-decoration:none;padding:9px 16px;border-radius:8px;font-weight:700;transition:background .2s;}
.colab-link:hover{background:#ffbf4d;}
```

- [ ] **Step 2: Convert the three class cards from anchors to divs with an action row**

Find:
```html
    <a class="class-card" href="class-1.html">
      <div class="class-num">01</div>
      <div class="class-body">
        <div class="class-title">What Is a Large Language Model?</div>
        <div class="class-desc">Next-token prediction, training at a high level, and the transformer/attention intuition behind modern LLMs.</div>
      </div>
      <div class="class-arrow">Start →</div>
    </a>
    <a class="class-card" href="class-2.html">
      <div class="class-num">02</div>
      <div class="class-body">
        <div class="class-title">Tokens, Temperature &amp; Context Windows</div>
        <div class="class-desc">Tokenization mechanics, temperature and top-p sampling, context window limits, and cost/latency tradeoffs.</div>
      </div>
      <div class="class-arrow">Start →</div>
    </a>
    <a class="class-card" href="class-3.html">
      <div class="class-num">03</div>
      <div class="class-body">
        <div class="class-title">Capabilities, Limitations &amp; Responsible AI</div>
        <div class="class-desc">Hallucination, bias, other failure modes, and practical principles for using LLMs responsibly.</div>
      </div>
      <div class="class-arrow">Start →</div>
    </a>
```
Replace with:
```html
    <div class="class-card">
      <div class="class-num">01</div>
      <div class="class-body">
        <div class="class-title">What Is a Large Language Model?</div>
        <div class="class-desc">Next-token prediction, training at a high level, and the transformer/attention intuition behind modern LLMs.</div>
        <div class="class-actions">
          <a class="class-arrow" href="class-1.html">Start →</a>
          <a class="colab-link" href="https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-1.ipynb" target="_blank" rel="noopener">▶ Open in Colab</a>
        </div>
      </div>
    </div>
    <div class="class-card">
      <div class="class-num">02</div>
      <div class="class-body">
        <div class="class-title">Tokens, Temperature &amp; Context Windows</div>
        <div class="class-desc">Tokenization mechanics, temperature and top-p sampling, context window limits, and cost/latency tradeoffs.</div>
        <div class="class-actions">
          <a class="class-arrow" href="class-2.html">Start →</a>
          <a class="colab-link" href="https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-2.ipynb" target="_blank" rel="noopener">▶ Open in Colab</a>
        </div>
      </div>
    </div>
    <div class="class-card">
      <div class="class-num">03</div>
      <div class="class-body">
        <div class="class-title">Capabilities, Limitations &amp; Responsible AI</div>
        <div class="class-desc">Hallucination, bias, other failure modes, and practical principles for using LLMs responsibly.</div>
        <div class="class-actions">
          <a class="class-arrow" href="class-3.html">Start →</a>
          <a class="colab-link" href="https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-3.ipynb" target="_blank" rel="noopener">▶ Open in Colab</a>
        </div>
      </div>
    </div>
```

- [ ] **Step 3: Verify the cards render correctly and links resolve**

Run:
```bash
python3 -c "
import re
html = open('week-2/index.html').read()
assert html.count('class=\"class-card\"') == 3, 'expected exactly 3 class cards'
assert '<a class=\"class-card\"' not in html, 'class-card should no longer be an anchor'
for n in (1, 2, 3):
    url = f'https://colab.research.google.com/github/spravesh1818/lfconnect-genai-b2/blob/main/week-2/class-{n}.ipynb'
    assert url in html, f'missing Colab link for class-{n}'
print('index.html structure verified')
"
```
Expected: `index.html structure verified`

Then open `week-2/index.html` in a browser and confirm: each card still shows title/description, hover still lifts the card, and both "Start →" and "▶ Open in Colab" are clickable and visually distinct (cyan outline vs. amber fill).

- [ ] **Step 4: Commit**

```bash
git add week-2/index.html
git commit -m "$(cat <<'EOF'
Wire Colab links into week-2 index, matching week-1's card pattern

Converts class cards from whole-card anchors to divs with a Start
link plus a Colab deep link, consistent with how week-1/index.html
already presents its class-N.ipynb notebooks.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Slide tone pass — `class-1.html`

**Files:**
- Modify: `week-2/class-1.html` (text inside existing `data-editable` elements only)

**Interfaces:**
- Consumes: nothing external. Constraint: exactly 13 `<section class="slide">` elements before and after; no attribute or tag changes.

### Voice guide (applies to Tasks 5, 6, and 7)

Rewrite each `data-editable` text node so it reads like a knowledgeable person explaining this to you, not marketing copy or a textbook abstract:

- Vary sentence length and openers — don't let every bullet start with the same "X does Y" shape.
- Break up the recurring "X — Y, not Z" / rule-of-three cadence; not every sentence needs an em-dash clause.
- Prefer contractions and direct address ("you'll notice", "here's the thing") over a formal, distant register — but keep it a teaching voice, not hype.
- Every technical claim must stay exactly as accurate as the original — this is a rhythm/voice edit, not a content edit. If in doubt about a claim's accuracy, leave the sentence's meaning untouched and only adjust phrasing.
- Titles/headings can loosen slightly (drop unnecessary ellipses, forced alliteration) but must still describe the slide's actual content.

**Worked examples (this deck):**
- `"It's Just... Next-Token Prediction"` → `"Under the Hood, It's Next-Token Prediction"`
- `"At each step, the model looks at all the text so far and asks: 'what token is most likely to come next?'"` → `"At every step the model looks back over everything written so far and asks one question: what's the most likely next token?"`
- `"There is no separate 'understanding module' — fluent-looking language emerges from this one repeated act."` → `"There's no separate 'understanding module' tucked away in there. Fluent language just falls out of doing this one small thing, over and over."`

- [ ] **Step 1: Snapshot the current structural skeleton**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-1.html').read()
skeleton = re.sub(r'>([^<]*)<', '><', html)
open('/tmp/class1-skeleton-before.txt', 'w').write(skeleton)
print('slide count:', html.count('<section class=\"slide\">'))
"
```
Expected: `slide count: 13`

- [ ] **Step 2: Rewrite the copy**

Using the Read tool, open `week-2/class-1.html` and use Edit to rewrite the text content of each `data-editable` element (and slide titles/subtitles) per the voice guide above. Do not touch any tag, class, attribute, or the `data-editable` markers themselves — only the text between tags.

- [ ] **Step 3: Verify structure is unchanged and slide count still matches**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-1.html').read()
skeleton_after = re.sub(r'>([^<]*)<', '><', html)
skeleton_before = open('/tmp/class1-skeleton-before.txt').read()
assert skeleton_after == skeleton_before, 'DOM structure changed — only text nodes should differ'
assert html.count('<section class=\"slide\">') == 13, 'slide count must stay 13'
print('structure unchanged, slide count still 13')
"
```
Expected: `structure unchanged, slide count still 13`

- [ ] **Step 4: Open in a browser and confirm no overflow**

Open `week-2/class-1.html` in a browser, step through all 13 slides (arrow keys), and confirm no text overflows its card/slide and nothing overlaps — rewritten text may run slightly longer or shorter than the original, so this needs an eyeball check even though the structure check passed.

- [ ] **Step 5: Commit**

```bash
git add week-2/class-1.html
git commit -m "$(cat <<'EOF'
Rewrite class-1 slide copy for a more human voice

Text-only pass inside existing data-editable nodes — same 13 slides,
same structure, same technical claims, less repetitive cadence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Slide tone pass — `class-2.html`

**Files:**
- Modify: `week-2/class-2.html` (text inside existing `data-editable` elements only)

**Interfaces:**
- Consumes: the voice guide defined in Task 5. Constraint: exactly 14 `<section class="slide">` elements before and after.

**Worked examples (this deck):**
- `"Models don't read letters or whole words — they read tokens: sub-word chunks produced by a tokenizer before any prediction happens."` → `"Before a model predicts anything, a tokenizer has already chopped your text into tokens — sub-word chunks, not letters or whole words."`
- `"Temperature reshapes the probability ranking before a token is sampled — it doesn't change what the model 'knows,' only how boldly it gambles."` → `"Temperature doesn't change what the model knows. It just changes how boldly it's willing to gamble when picking the next token."`

- [ ] **Step 1: Snapshot the current structural skeleton**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-2.html').read()
skeleton = re.sub(r'>([^<]*)<', '><', html)
open('/tmp/class2-skeleton-before.txt', 'w').write(skeleton)
print('slide count:', html.count('<section class=\"slide\">'))
"
```
Expected: `slide count: 14`

- [ ] **Step 2: Rewrite the copy**

Using the Read tool, open `week-2/class-2.html` and use Edit to rewrite the text content of each `data-editable` element per the Task 5 voice guide. Do not touch any tag, class, or attribute.

- [ ] **Step 3: Verify structure is unchanged and slide count still matches**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-2.html').read()
skeleton_after = re.sub(r'>([^<]*)<', '><', html)
skeleton_before = open('/tmp/class2-skeleton-before.txt').read()
assert skeleton_after == skeleton_before, 'DOM structure changed — only text nodes should differ'
assert html.count('<section class=\"slide\">') == 14, 'slide count must stay 14'
print('structure unchanged, slide count still 14')
"
```
Expected: `structure unchanged, slide count still 14`

- [ ] **Step 4: Open in a browser and confirm no overflow**

Open `week-2/class-2.html` in a browser, step through all 14 slides, confirm no overflow/overlap, and confirm the temperature gauge and probability-bar slides (which have precise numeric captions) still read correctly.

- [ ] **Step 5: Commit**

```bash
git add week-2/class-2.html
git commit -m "$(cat <<'EOF'
Rewrite class-2 slide copy for a more human voice

Text-only pass inside existing data-editable nodes — same 14 slides,
same structure, same technical claims, less repetitive cadence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Slide tone pass — `class-3.html`

**Files:**
- Modify: `week-2/class-3.html` (text inside existing `data-editable` elements only)

**Interfaces:**
- Consumes: the voice guide defined in Task 5. Constraint: exactly 13 `<section class="slide">` elements before and after.

**Worked examples (this deck):**
- `"A hallucination is fluent, plausible-sounding output that is factually wrong or entirely invented — stated with the same confidence as a correct answer."` → `"A hallucination is output that sounds completely plausible and is simply wrong, or made up outright — and it's said with exactly the same confidence as a true answer."`
- `"None of these limitations mean 'don't use LLMs' — they mean use them with the right safeguards."` → `"None of this is an argument against using LLMs. It's an argument for using them with the right safeguards in place."`

- [ ] **Step 1: Snapshot the current structural skeleton**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-3.html').read()
skeleton = re.sub(r'>([^<]*)<', '><', html)
open('/tmp/class3-skeleton-before.txt', 'w').write(skeleton)
print('slide count:', html.count('<section class=\"slide\">'))
"
```
Expected: `slide count: 13`

- [ ] **Step 2: Rewrite the copy**

Using the Read tool, open `week-2/class-3.html` and use Edit to rewrite the text content of each `data-editable` element per the Task 5 voice guide. Do not touch any tag, class, or attribute.

- [ ] **Step 3: Verify structure is unchanged and slide count still matches**

Run:
```bash
python3 -c "
import re
html = open('week-2/class-3.html').read()
skeleton_after = re.sub(r'>([^<]*)<', '><', html)
skeleton_before = open('/tmp/class3-skeleton-before.txt').read()
assert skeleton_after == skeleton_before, 'DOM structure changed — only text nodes should differ'
assert html.count('<section class=\"slide\">') == 13, 'slide count must stay 13'
print('structure unchanged, slide count still 13')
"
```
Expected: `structure unchanged, slide count still 13`

- [ ] **Step 4: Open in a browser and confirm no overflow**

Open `week-2/class-3.html` in a browser, step through all 13 slides, confirm no overflow/overlap, and confirm the hallucination example box still reads as a coherent prompt/model/reality sequence.

- [ ] **Step 5: Commit**

```bash
git add week-2/class-3.html
git commit -m "$(cat <<'EOF'
Rewrite class-3 slide copy for a more human voice

Text-only pass inside existing data-editable nodes — same 13 slides,
same structure, same technical claims, less repetitive cadence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
