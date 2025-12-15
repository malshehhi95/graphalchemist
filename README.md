# Graph Alchemist 🧪📊

Turn chart images into structured data and generate intelligent alternative visualizations using Vision + LLM reasoning.


🚀 **Live Demo**  
👉 https://graphalchemist.onrender.com


![alt text](https://github.com/malshehhi95/graphalchemist/blob/main/poster.png?raw=true)


## How it works (Technical Methodology)

Graph Alchemist follows a multi-stage, controlled pipeline that combines
computer vision, large language model reasoning, and deterministic code execution.
Each stage has a clear responsibility and explicit validation to avoid hallucination
and semantic drift.

### 1. Chart Image Ingestion
The system starts by accepting an image of a chart (e.g. bar chart, categorical plot).
No assumptions are made about the original plotting library or data source.

### 2. Vision-Based Data Extraction
A multimodal vision-language model (Qwen-VL) analyzes the image and extracts
structured data into a strict JSON schema:

- chart type (if inferable)
- axis labels
- category labels
- numeric values
- uncertainty notes (if any estimation was required)

This stage focuses only on **seeing**, not reasoning or visualization choices.
If uncertainty exists, it is explicitly recorded instead of hidden.

### 3. Validation and Human-in-the-Loop Gate
The extracted JSON is validated for structural consistency
(e.g. category/value alignment).

If the vision model reports uncertainty, the system pauses and asks the user
to confirm or edit the extracted data before proceeding.
This prevents silent propagation of incorrect assumptions.

### 4. Visualization Strategy Selection
Before generating any plots, the system selects the most suitable visualization
styles based on the data characteristics:

- number of categories
- value distribution and spread
- presence of negative values
- readability constraints

Only semantically appropriate chart types are considered.
The selection logic is deterministic and runs in Python, not inside the LLM.

### 5. Reasoning + Code Generation
A large language model (LLaMA 3.3 70B via Groq) is then used **only** for:
- reasoning about *why* each selected visualization is suitable
- generating the corresponding Matplotlib code
- explaining the scenario where each chart is useful

The model is constrained to:
- produce raw Python code only (no markdown)
- respect axis semantics
- use a limited, sandboxed execution environment

Each visualization is returned as a structured object containing:
- the chart style identifier
- a short usage scenario
- a justification (“why this chart fits”)
- optional caveats
- the exact plotting code

### 6. Safe Local Execution
Generated code is sanitized and executed locally in a restricted environment:
- no file I/O
- no network access
- limited builtins
- controlled libraries (Matplotlib + NumPy only)

This guarantees deterministic rendering and prevents arbitrary execution.

### 7. Transparent Output
For each generated visualization, the system outputs:
- the rendered image
- an explanation of why the chart is suitable
- the exact code used to generate it (revealed on demand)

This makes the system auditable, educational, and reproducible.

---

**Key Design Principle:**  
The system does not “redraw charts”.
It **understands data, selects representations intentionally,
and explains its choices**, combining AI reasoning with strict engineering control.
