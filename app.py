import os
import json
import base64
import re
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI
from groq import Groq


DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_data_url(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    # Good enough for jpg/png in practice for these endpoints
    return f"data:image/png;base64,{b64}"


def extract_first_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))


def vision_extract_chart(image_path: Path) -> dict:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY env var.")

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)

    schema = {
        "chart_type": "bar",
        "title": "",
        "x_label": "",
        "y_label": "",
        "categories": [],
        "values": [],
        "notes": ""
    }

    instruction = f"""
You are a chart data extraction engine.

Return ONLY valid JSON. No markdown fences, no extra text.
If you add any text outside JSON, the program will fail.

Use exactly this JSON structure (same keys):
{json.dumps(schema, ensure_ascii=False)}

Rules:
- categories must be a list of strings in x-axis order.
- values must be a list of numbers aligned with categories.
- Fill x_label and y_label if visible.
- notes: if anything is unclear or estimated, write it here. Otherwise empty string.
"""

    img_url = to_data_url(image_path)

    resp = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    data = extract_first_json(raw)

    if len(data.get("categories", [])) != len(data.get("values", [])):
        raise ValueError("Vision JSON invalid: categories and values length mismatch.")

    return data


def pick_styles(data: dict, min_k: int = 3, max_k: int = 3) -> list[str]:
    cats = data.get("categories", [])
    vals = data.get("values", [])
    n = len(cats)
    has_neg = any(v < 0 for v in vals) if vals else False

    vmin = min(vals) if vals else 0
    vmax = max(vals) if vals else 0
    spread = vmax - vmin

    candidates = ["barh_gradient", "highlight_max_bar"]

    if n <= 6 and not has_neg:
        candidates.append("donut")
    else:
        candidates.append("dot_ranking")

    # Optional extra style if user asks for 4
    if (spread > 0) and (n <= 10):
        candidates.append("lollipop")

    unique = []
    for s in candidates:
        if s not in unique:
            unique.append(s)

    if len(unique) < min_k:
        unique.append("dot_ranking")
        unique = list(dict.fromkeys(unique))

    k = min(max_k, max(min_k, len(unique)))
    return unique[:k]


def groq_generate_items(data: dict, styles: list[str]) -> list[dict]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY env var.")

    client = Groq(api_key=api_key)

    expected = len(styles)
    style_list = ", ".join(styles)

    schema = {
        "items": [
            {
                "style_id": "barh_gradient",
                "scenario": "When to use this chart (1-2 sentences).",
                "why": "Why it fits this specific data (short).",
                "notes": "Limitations/caveats (optional).",
                "code": "Python code as a single string with \\n newlines."
            }
        ]
    }

    prompt = f"""
Return ONLY valid JSON. No markdown. No triple backticks.

Generate EXACTLY {expected} items, in this exact order of style_id:
{style_list}

JSON schema (use same keys):
{json.dumps(schema, ensure_ascii=False)}

Rules for each item's code:
- Must be matplotlib code only
- Must create exactly one figure
- Must plot the same data
- Must not call plt.show()
- Must not read files, no network, no seaborn
- You MAY use numpy as np for spacing/indexing only
- Axis semantics:
  * categories on Y, values on X => xlabel=data["y_label"], ylabel=data["x_label"]
  * categories on X, values on Y => xlabel=data["x_label"], ylabel=data["y_label"]
  * donut => do not set x/y labels

Style ID specs (follow exactly):
- barh_gradient: horizontal bar chart, categories on Y, values on X, colormap based on magnitude, value labels.
- lollipop: categories on Y, values on X, thin line from 0 to value + dot at value, value labels.
- dot_ranking: categories on Y, values on X, dots only, gridlines, value labels.
- donut: donut chart with percentages, raw values in legend or labels, no x/y labels.
- highlight_max_bar: vertical bar chart, max bar highlighted, others muted, annotate max.
"""

    msg = prompt + "\n\nDATA:\n" + json.dumps(data, ensure_ascii=False)

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": msg}],
        temperature=0.6
    )

    raw = resp.choices[0].message.content
    payload = extract_first_json(raw)

    items = payload.get("items", [])
    if len(items) != expected:
        raise ValueError(f"Expected {expected} items, got {len(items)}")

    for i, sid in enumerate(styles):
        if items[i].get("style_id") != sid:
            raise ValueError(f"Item #{i+1} style_id mismatch: expected {sid}, got {items[i].get('style_id')}")

    return items


def clean_llm_code(code: str) -> str:
    code = (code or "").strip()

    # remove markdown fences if they appear
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
    code = code.replace("```", "")

    # strip only the "allowed" imports if present
    lines = []
    for line in code.splitlines():
        s = line.strip()
        if s in ("import matplotlib.pyplot as plt", "import numpy as np"):
            continue
        if s.startswith("from matplotlib") or s.startswith("import matplotlib"):
            continue
        if s.startswith("import numpy"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def run_block_and_save(code: str, data: dict, out_path: Path):
    plt.close("all")

    code = clean_llm_code(code)

    banned = ["open(", "import os", "import sys", "subprocess", "requests", "http", "socket", "pathlib"]
    for b in banned:
        if b in code:
            raise ValueError(f"Banned token found in code: {b}")

    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "sum": sum,
        "float": float, "int": int, "str": str, "list": list, "dict": dict,
        "enumerate": enumerate, "zip": zip
    }

    env = {
        "__builtins__": safe_builtins,
        "plt": plt,
        "np": np,
        "data": data
    }

    exec(code, env, env)

    fig = plt.gcf()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def build_results(job_id: str, image_path: Path, data: dict, k: int):
    # Ensure title exists
    if not data.get("title"):
        data["title"] = "Chart Alternative Views"

    styles = pick_styles(data, min_k=min(3, k), max_k=k)
    items = groq_generate_items(data, styles)

    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, item in enumerate(items, start=1):
        fname = f"graph_{idx}_{item['style_id']}.png"
        out_path = job_dir / fname
        run_block_and_save(item["code"], data, out_path)

        cleaned = clean_llm_code(item["code"])
        display_code = (
            "# NOTE: This snippet expects a dict named `data` with keys:\n"
            "# data['categories'], data['values'], data['x_label'], data['y_label'], data['title']\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            f"{cleaned}\n"
        )

        results.append({
            "style_id": item["style_id"],
            "scenario": item.get("scenario", ""),
            "why": item.get("why", ""),
            "notes": item.get("notes", ""),
            "img_url": url_for("static", filename=f"results/{job_id}/{fname}"),
            "code": display_code


        })

    return results, styles


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")


@app.get("/")
def home():
    return redirect(url_for("upload_form"))


@app.get("/upload")
def upload_form():
    return render_template("upload.html")


@app.post("/upload")
def upload_post():
    f = request.files.get("image")
    if not f or not f.filename:
        flash("Please upload an image file.")
        return redirect(url_for("upload_form"))

    k = int(request.form.get("k", "3"))
    k = 4 if k == 4 else 3

    job_id = uuid.uuid4().hex
    img_path = UPLOAD_DIR / f"{job_id}_{f.filename}"
    f.save(img_path)

    try:
        data = vision_extract_chart(img_path)
    except Exception as e:
        flash(f"Vision extraction failed: {e}")
        return redirect(url_for("upload_form"))

    # Save JSON so confirm step can load it
    (UPLOAD_DIR / f"{job_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # If vision notes exist, ask for confirmation/edit
    notes = (data.get("notes") or "").strip()
    if notes:
        return render_template(
            "confirm.html",
            job_id=job_id,
            image_url=None,
            json_text=json.dumps(data, ensure_ascii=False, indent=2),
            notes=notes,
            k=k
        )

    # Otherwise proceed directly
    try:
        results, styles = build_results(job_id, img_path, data, k)
    except Exception as e:
        flash(f"Graph generation failed: {e}")
        return redirect(url_for("upload_form"))

    return render_template("results.html", job_id=job_id, data=data, results=results, styles=styles)


@app.post("/confirm/<job_id>")
def confirm_post(job_id: str):
    k = int(request.form.get("k", "3"))
    k = 4 if k == 4 else 3

    json_text = request.form.get("json_text", "")
    try:
        data = json.loads(json_text)
    except Exception as e:
        flash(f"Invalid JSON: {e}")
        return redirect(url_for("upload_form"))

    # Find original uploaded image by prefix
    candidates = list(UPLOAD_DIR.glob(f"{job_id}_*"))
    if not candidates:
        flash("Original uploaded image not found.")
        return redirect(url_for("upload_form"))
    img_path = candidates[0]

    try:
        results, styles = build_results(job_id, img_path, data, k)
    except Exception as e:
        flash(f"Graph generation failed: {e}")
        return redirect(url_for("upload_form"))

    return render_template("results.html", job_id=job_id, data=data, results=results, styles=styles)


if __name__ == "__main__":
    app.run(debug=True)
