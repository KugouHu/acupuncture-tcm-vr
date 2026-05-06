"""
TCM Facial Diagnosis Server
---------------------------
FastAPI service that:
  1. Accepts a facial image upload
  2. Detects face landmarks (Haar + MediaPipe FaceMesh)
  3. Maps five facial regions to the Five Zang Organs and Five-Element colors
  4. Calls a multimodal LLM (Qwen-VL) for observational text
  5. Returns structured JSON + base64 images consumed by the Unity VR client
"""

import base64
import io
import json
import os
import re
import time
import uuid
from typing import Any, Dict

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter

# ==========================
# FastAPI app & CORS
# ==========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store (swap for a database in production).
tasks: Dict[str, Dict[str, Any]] = {}

# ==========================
# Static TCM intro text
# ==========================

TCM_INTRO = (
    "In traditional Chinese medicine, the Five Elements (metal, wood, water, fire, and earth) "
    "and the Five Colors (yellow, green, white, red, and black) are associated with the body's "
    "five organs (heart, liver, spleen, lungs, and kidneys), and it is believed that the condition "
    "of these organs is reflected in different regions of the face. "
    "The brightness and redness of the forehead generally reflect the strength of heart qi, "
    "the darkness or yellowish tone between the eyebrows can indicate whether liver qi is flowing smoothly, "
    "changes in the yellowish or dull color along the nose bridge are often related to spleen and digestive function, "
    "the paleness, graying, or flushing of the cheeks may reflect the state of lung qi and skin defense, "
    "and the dullness or lack of vitality in the chin area is commonly associated with kidney qi and overall energy levels."
)

# ==========================
# Qwen-VL client (OpenAI-compatible mode)
# ==========================
# Set DASHSCOPE_API_KEY in your environment before starting the server.

qwen_client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==========================
# Five-organ color configuration
# Keys are kept in Chinese because the rendering / analysis logic
# references them throughout this module.
# ==========================

ORGAN_COLOR_CONFIG = {
    "心": {  # heart / fire
        "A": {"element": "火", "level_desc": "balanced",        "tone_desc": "soft rosy red",                "rgb": (244, 163, 163)},
        "B": {"element": "火", "level_desc": "mild deviation",  "tone_desc": "moderately reddish",           "rgb": (242, 106, 106)},
        "C": {"element": "火", "level_desc": "moderate",        "tone_desc": "saturated red",                "rgb": (216,  52,  52)},
        "D": {"element": "火", "level_desc": "pronounced",      "tone_desc": "deep red, very prominent",     "rgb": (156,  28,  28)},
    },
    "肝": {  # liver / wood
        "A": {"element": "木", "level_desc": "balanced",        "tone_desc": "pale green, soft",             "rgb": (168, 213, 186)},
        "B": {"element": "木", "level_desc": "mild deviation",  "tone_desc": "noticeably greenish",          "rgb": (107, 191,  89)},
        "C": {"element": "木", "level_desc": "moderate",        "tone_desc": "saturated green",              "rgb": ( 63, 145,  66)},
        "D": {"element": "木", "level_desc": "pronounced",      "tone_desc": "deep green, very prominent",   "rgb": ( 30,  86,  49)},
    },
    "脾": {  # spleen / earth
        "A": {"element": "土", "level_desc": "balanced",        "tone_desc": "soft pale yellow",             "rgb": (247, 227, 163)},
        "B": {"element": "土", "level_desc": "mild deviation",  "tone_desc": "noticeably yellow",            "rgb": (242, 204,  92)},
        "C": {"element": "土", "level_desc": "moderate",        "tone_desc": "deep saturated yellow",        "rgb": (224, 168,   0)},
        "D": {"element": "土", "level_desc": "pronounced",      "tone_desc": "very deep yellow",             "rgb": (181, 129,   0)},
    },
    "肺": {  # lung / metal
        "A": {"element": "金", "level_desc": "balanced",        "tone_desc": "near-white, soft and bright",  "rgb": (249, 249, 249)},
        "B": {"element": "金", "level_desc": "mild deviation",  "tone_desc": "light gray-white",             "rgb": (229, 229, 229)},
        "C": {"element": "金", "level_desc": "moderate",        "tone_desc": "medium gray-white",            "rgb": (204, 204, 204)},
        "D": {"element": "金", "level_desc": "pronounced",      "tone_desc": "darker gray-white",            "rgb": (179, 179, 179)},
    },
    "肾": {  # kidney / water
        "A": {"element": "水", "level_desc": "balanced",        "tone_desc": "soft dark gray-blue",          "rgb": (160, 164, 184)},
        "B": {"element": "水", "level_desc": "mild deviation",  "tone_desc": "deeper gray-blue",             "rgb": (107, 114, 128)},
        "C": {"element": "水", "level_desc": "moderate",        "tone_desc": "saturated dark gray-blue",     "rgb": ( 55,  65,  81)},
        "D": {"element": "水", "level_desc": "pronounced",      "tone_desc": "near-black, very concentrated", "rgb": ( 17,  24,  39)},
    },
}

ORG_TAGS = ["Heart", "Liver", "Spleen", "Lung", "Kidney"]


# ==========================
# Helpers
# ==========================

def downscale_image_half(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_small = img.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
    buf = io.BytesIO()
    img_small.save(buf, format="JPEG")
    return buf.getvalue()


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def data_url_from_bytes(image_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + image_bytes_to_base64(image_bytes)


def extract_organ_sections(text: str):
    """Split LLM output into one entry per organ tag; always returns 5 entries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    bucket = {tag: [] for tag in ORG_TAGS}

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        for tag in ORG_TAGS:
            if re.search(rf"\b{tag}\b", s, flags=re.IGNORECASE):
                bucket[tag].append(s)
                break

    return [{"tag": tag, "text": " ".join(bucket[tag]).strip()} for tag in ORG_TAGS]


def split_model_analysis_by_orgs(text: str) -> Dict[str, str]:
    """Slice the LLM analysis into organ-keyed segments by first occurrence."""
    if not text:
        return {}

    positions = []
    for tag in ORG_TAGS:
        m = re.search(rf"\b{tag}\b", text, flags=re.IGNORECASE)
        if m:
            positions.append((m.start(), tag))

    if not positions:
        return {"all": text.strip()}

    positions.sort(key=lambda x: x[0])
    result: Dict[str, str] = {}
    for i, (start, tag) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        result[tag.lower()] = text[start:end].strip()
    return result


# ==========================
# Color-based five-organ analysis (Haar + region averages)
# ==========================

def analyze_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        fx, fy, fw, fh = 0, 0, w, h
    else:
        fx, fy, fw, fh = faces[0]

    fx, fy = max(0, fx), max(0, fy)
    fw, fh = min(fw, w - fx), min(fh, h - fy)

    def roi_rect(rel_x0, rel_y0, rel_x1, rel_y1):
        x0 = max(0, int(fx + rel_x0 * fw))
        y0 = max(0, int(fy + rel_y0 * fh))
        x1 = min(w, int(fx + rel_x1 * fw))
        y1 = min(h, int(fy + rel_y1 * fh))
        return x0, y0, x1, y1

    organ_rois = {
        "心": [roi_rect(0.22, 0.00, 0.78, 0.25)],                                 # forehead
        "肝": [roi_rect(0.40, 0.25, 0.60, 0.40)],                                 # glabella
        "脾": [roi_rect(0.47, 0.35, 0.53, 0.68)],                                 # nose bridge
        "肺": [roi_rect(0.06, 0.35, 0.40, 0.75), roi_rect(0.60, 0.35, 0.94, 0.75)],  # cheeks
        "肾": [roi_rect(0.28, 0.70, 0.72, 0.96)],                                 # chin
    }

    organ_avg_rgb = {}
    for organ, boxes in organ_rois.items():
        pixels = []
        for (x0, y0, x1, y1) in boxes:
            if x1 > x0 and y1 > y0:
                roi = img_np[y0:y1, x0:x1, :]
                if roi.size > 0:
                    pixels.append(roi.reshape(-1, 3))
        organ_avg_rgb[organ] = tuple(np.vstack(pixels).mean(axis=0).tolist()) if pixels else (0, 0, 0)

    def closest_level(organ, rgb):
        r, g, b = rgb
        best_level, best_dist = "A", float("inf")
        for level, cfg in ORGAN_COLOR_CONFIG[organ].items():
            cr, cg, cb = cfg["rgb"]
            dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if dist < best_dist:
                best_dist, best_level = dist, level
        return best_level

    organ_states = {organ: closest_level(organ, rgb) for organ, rgb in organ_avg_rgb.items()}
    organ_rgb = {organ: ORGAN_COLOR_CONFIG[organ][level]["rgb"] for organ, level in organ_states.items()}

    parts = []
    for organ, level in organ_states.items():
        cfg = ORGAN_COLOR_CONFIG[organ][level]
        parts.append(f"{organ}: {cfg['level_desc']}, tone '{cfg['tone_desc']}'")
    summary = (
        "Color-based observation across facial regions: "
        + "; ".join(parts)
        + ". This is a visualization-only summary, not medical advice."
    )

    return organ_states, organ_rgb, summary


# ==========================
# Qwen-VL multimodal analysis
# ==========================

def analyze_image_with_qwen(image_bytes: bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    completion = qwen_client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the facial regions that correspond to the five organs "
                            "in traditional Chinese medicine (heart, liver, spleen, lung, kidney), "
                            "based purely on visual observation of color and brightness. "
                            "Describe your observations in English and avoid any medical diagnosis or treatment advice."
                        ),
                    },
                ],
            }
        ],
    )

    content = completion.choices[0].message.content
    if isinstance(content, list):
        analysis_text = "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    else:
        analysis_text = str(content).strip()

    analysis_text += (
        "\n\n(This description is generated by an AI model based on image analysis, "
        "for reference only and not medical advice.)"
    )

    organ_states_placeholder = {organ: "B" for organ in ORGAN_COLOR_CONFIG}
    return organ_states_placeholder, analysis_text


# ==========================
# FaceMesh-based color overlay rendering
# ==========================

def render_organ_images(image_bytes: bytes, organ_states: Dict[str, str]) -> Dict[str, bytes]:
    """
    Render six images: a combined overlay plus one per organ.
    Each region is painted with its Five-Element color and Gaussian-blurred for
    a soft mask consistent with the VR client visuals.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    if not results.multi_face_landmarks:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b = buf.getvalue()
        return {key: b for key in ("combined", "heart", "liver", "spleen", "lung", "kidney")}

    face = results.multi_face_landmarks[0]
    pts = [(p.x * w, p.y * h) for p in face.landmark]

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)

    forehead_center = pts[10]
    glabella_center = pts[9]
    nose_center     = pts[6]
    left_cheek      = pts[234]
    right_cheek     = pts[454]
    chin_center     = pts[152]

    def draw_for_orgs(org_list):
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def add_ellipse(organ, center_xy, rx_scale, ry_scale):
            SCALE = 0.9
            cx, cy = center_xy
            rx, ry = fw * rx_scale * SCALE, fh * ry_scale * SCALE
            x0, y0 = max(0, cx - rx), max(0, cy - ry)
            x1, y1 = min(w, cx + rx), min(h, cy + ry)
            level = organ_states.get(organ, "B")
            r, g, b = ORGAN_COLOR_CONFIG[organ][level]["rgb"]
            draw.ellipse([x0, y0, x1, y1], fill=(r, g, b, 120))

        if "心" in org_list:
            add_ellipse("心", (forehead_center[0], forehead_center[1] - 0.10 * fh), 0.30, 0.18)
        if "肝" in org_list:
            add_ellipse("肝", glabella_center, 0.10, 0.07)
        if "脾" in org_list:
            spleen_center = ((glabella_center[0] + nose_center[0]) / 2,
                             (glabella_center[1] + nose_center[1]) / 2)
            add_ellipse("脾", spleen_center, 0.07, 0.20)
        if "肺" in org_list:
            add_ellipse("肺", left_cheek,  0.18, 0.24)
            add_ellipse("肺", right_cheek, 0.18, 0.24)
        if "肾" in org_list:
            add_ellipse("肾", (chin_center[0], chin_center[1] - 0.05 * fh), 0.20, 0.15)

        overlay_blur = overlay.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.015)))
        combined_img = Image.alpha_composite(img.convert("RGBA"), overlay_blur).convert("RGB")
        buf = io.BytesIO()
        combined_img.save(buf, format="JPEG")
        return buf.getvalue()

    return {
        "combined": draw_for_orgs(["心", "肝", "脾", "肺", "肾"]),
        "heart":    draw_for_orgs(["心"]),
        "liver":    draw_for_orgs(["肝"]),
        "spleen":   draw_for_orgs(["脾"]),
        "lung":     draw_for_orgs(["肺"]),
        "kidney":   draw_for_orgs(["肾"]),
    }


# ==========================
# Endpoints
# ==========================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return {"success": False, "error": "Empty file"}

        image_bytes = downscale_image_half(image_bytes)

        organ_states, organ_rgb, color_summary = analyze_image(image_bytes)
        _, model_analysis = analyze_image_with_qwen(image_bytes)
        organ_tags = extract_organ_sections(model_analysis)

        organ_images = render_organ_images(image_bytes, organ_states)
        combined_bytes = organ_images["combined"]

        original_b64 = image_bytes_to_base64(image_bytes)
        combined_b64 = image_bytes_to_base64(combined_bytes)

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "original_b64": original_b64,
            "combined_b64": combined_b64,
            "organ_states": organ_states,
            "organ_rgb": organ_rgb,
            "color_summary": color_summary,
            "model_analysis": model_analysis,
            "organ_tags": organ_tags,
            "timestamp": time.time(),
        }

        os.makedirs("results", exist_ok=True)
        json_path = f"results/{task_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(jsonable_encoder(tasks[task_id]), f, ensure_ascii=False, indent=2)
        tasks[task_id]["json_path"] = json_path

        return {
            "success": True,
            "task_id": task_id,
            "organ_states": organ_states,
            "organ_rgb": organ_rgb,
            "color_summary": color_summary,
            "model_analysis": model_analysis,
            "organ_tags": organ_tags,
            "original_base64": original_b64,
            "combined_base64": combined_b64,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


VALID_IMAGE_TYPES = {k: k for k in ("original", "combined", "heart", "liver", "spleen", "lung", "kidney")}


@app.get("/image/{task_id}/{kind}")
async def get_image(task_id: str, kind: str):
    kind = kind.lower()
    if kind not in VALID_IMAGE_TYPES:
        return JSONResponse(status_code=400, content={"success": False, "error": "invalid image type"})
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"success": False, "error": "task_id not found"})

    img_bytes = tasks[task_id].get(VALID_IMAGE_TYPES[kind])
    if img_bytes is None:
        return JSONResponse(status_code=404, content={"success": False, "error": "image not available"})
    return Response(content=img_bytes, media_type="image/jpeg")


@app.get("/latest")
async def latest():
    if not tasks:
        return {"success": False, "error": "No analysis available"}

    latest_task_id = max(tasks, key=lambda tid: tasks[tid].get("timestamp", 0))
    item = tasks.get(latest_task_id, {})
    return {
        "success": True,
        "task_id": latest_task_id,
        "organ_states": item.get("organ_states", {}),
        "organ_rgb": item.get("organ_rgb", {}),
        "color_summary": item.get("color_summary", ""),
        "model_analysis": item.get("model_analysis", ""),
        "organ_tags": item.get("organ_tags", []),
        "tcm_intro": TCM_INTRO,
        "original_base64": item.get("original_b64", ""),
        "combined_base64": item.get("combined_b64", ""),
    }


@app.get("/latest_json")
async def latest_json():
    if not tasks:
        return {"success": False, "error": "No analysis yet"}
    latest_task_id = max(tasks, key=lambda tid: tasks[tid]["timestamp"])
    json_path = tasks[latest_task_id]["json_path"]
    return FileResponse(json_path, filename=f"{latest_task_id}.json", media_type="application/json")


# ==========================
# Minimal upload UI (Device A)
# ==========================

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
      <meta charset="utf-8" />
      <title>TCM Face Analysis - Device A</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; }
        h1 { font-size: 24px; }
        .section { margin-bottom: 24px; }
        input[type="file"] { margin-top: 8px; }
        button { margin-top: 8px; padding: 6px 12px; }
        code { background:#f5f5f5; padding:2px 4px; border-radius:4px; }
        a { word-break: break-all; }
      </style>
    </head>
    <body>
      <h1>TCM Face Analysis - Upload (Device A)</h1>

      <div class="section">
        <h2>1. Upload an Image</h2>
        <input type="file" id="fileInput" accept="image/*" />
        <br/>
        <button onclick="upload()">Upload &amp; Analyze</button>
        <p id="uploadMsg"></p>
        <p><b>task_id:</b> <span id="taskId"></span></p>
      </div>

      <div class="section">
        <h2>2. Image URLs for this task</h2>
        <ul id="imageLinks"></ul>
      </div>

      <div class="section">
        <h2>Preview Images</h2>
        <div id="preview" style="display:flex;flex-direction:row;"></div>
      </div>

      <div class="section">
        <h2>3. Latest Result (Device B)</h2>
        <p>You can also check <code>/latest</code> for the newest task.</p>
        <button onclick="window.open('/latest','_blank')">Open /latest in new tab</button>
      </div>

      <script>
      async function upload() {
        const fileInput = document.getElementById('fileInput');
        const msg = document.getElementById('uploadMsg');
        const tidSpan = document.getElementById('taskId');
        const linksUl = document.getElementById('imageLinks');
        const previewDiv = document.getElementById('preview');

        msg.innerText = '';
        tidSpan.innerText = '';
        linksUl.innerHTML = '';
        previewDiv.innerHTML = '';

        if (!fileInput.files || fileInput.files.length === 0) {
          alert('Please choose an image first.');
          return;
        }

        const fd = new FormData();
        fd.append('file', fileInput.files[0]);

        try {
          const res = await fetch('/upload', { method: 'POST', body: fd });
          const data = await res.json();
          if (!data.success) {
            msg.innerText = 'Upload failed: ' + (data.error || 'unknown error');
            return;
          }
          msg.innerText = 'Upload success.';
          tidSpan.innerText = data.task_id;

          const originalImg = document.createElement('img');
          originalImg.src = "data:image/jpeg;base64," + data.original_base64;
          originalImg.style.width = "200px";
          originalImg.style.marginRight = "20px";
          const combinedImg = document.createElement('img');
          combinedImg.src = "data:image/jpeg;base64," + data.combined_base64;
          combinedImg.style.width = "200px";
          previewDiv.appendChild(originalImg);
          previewDiv.appendChild(combinedImg);

          const base = window.location.origin;
          for (const k of ['original', 'combined']) {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `${base}/image/${data.task_id}/${k}`;
            a.target = '_blank';
            a.innerText = `${k} -> ${a.href}`;
            li.appendChild(a);
            linksUl.appendChild(li);
          }
        } catch (e) {
          console.error(e);
          msg.innerText = 'Upload error: ' + e;
        }
      }
      </script>
    </body>
    </html>
    """
