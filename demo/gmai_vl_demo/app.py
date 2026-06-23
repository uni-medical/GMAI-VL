import os
import time
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


DEMO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = DEMO_ROOT / "model_weight"
DEFAULT_GRADIO_TEMP_DIR = DEMO_ROOT / "gradio_tmp"

MODEL_PATH = os.environ.get(
    "GMAI_VL_MODEL_PATH",
    str(DEFAULT_MODEL_PATH),
)
SAMPLE_DIR = DEMO_ROOT / "sample_assets"
os.environ.setdefault("GRADIO_TEMP_DIR", str(DEFAULT_GRADIO_TEMP_DIR))
Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)
HERO_IMAGE = DEMO_ROOT / "static" / "hero-medical-ai.png"
SAMPLE_PROMPTS = SAMPLE_DIR / "sample_prompts.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


TASK_PROMPTS: Dict[str, str] = {
    "Caption": "Describe the medical image concisely, including modality, visible anatomy, and key abnormal or normal findings.",
    "VQA": "Answer this medical visual question: What are the most relevant findings in this image?",
    "Report": "Write a short radiology-style impression with the most important observations first.",
    "Localization": "Identify the likely region of interest and explain what visual evidence supports it.",
    "Differential": "Provide a ranked differential diagnosis and mention what visual clues support each option.",
    "Safety": "Flag any urgent or safety-critical findings that would require immediate clinical attention.",
}


BODY_SYSTEMS = [
    "Chest / Lung",
    "Brain / Neuro",
    "Abdomen",
    "Breast",
    "Eye / Retina",
    "Pathology",
    "Dermatology",
    "Skin / Dermoscopy",
    "Gastrointestinal / Endoscopy",
    "MRI / Neuro",
    "Musculoskeletal",
]


@dataclass
class ModelBundle:
    tokenizer: object
    image_processor: object
    model: object
    device: str
    dtype: torch.dtype


_BUNDLE: Optional[ModelBundle] = None


def _sample_meta() -> Dict[str, Dict[str, str]]:
    if SAMPLE_PROMPTS.exists():
        with SAMPLE_PROMPTS.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _label_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def _find_sample_file(label: str, meta: Dict[str, str]) -> Optional[Path]:
    if meta.get("file"):
        path = SAMPLE_DIR / meta["file"]
        if path.exists():
            return path

    aliases = {
        "Chest X-ray": "chest_xray",
        "Abdomen CT": "abdomen_ct",
        "Retina": "retina",
        "Pathology": "pathology",
        "Dermoscopy": "dermoscopy",
        "Endoscopy": "endoscopy",
        "MR": "MR",
    }
    candidates = [aliases.get(label, label), label]
    normalized = {candidate.lower().replace(" ", "_").replace("-", "_") for candidate in candidates}
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.lower().replace("-", "_") in normalized:
            return path
    return None


def _ensure_default_samples() -> None:
    SAMPLE_DIR.mkdir(exist_ok=True)
    specs = [
        ("chest_xray.png", "Chest X-ray", (28, 35, 42), "LUNGS"),
        ("abdomen_ct.png", "Abdomen CT", (34, 34, 38), "CT"),
        ("retina.png", "Retina", (42, 20, 18), "FUNDUS"),
        ("pathology.png", "Pathology", (246, 228, 226), "H&E"),
    ]
    for name, _label, bg, mark in specs:
        path = SAMPLE_DIR / name
        if not path.exists():
            img = Image.new("RGB", (640, 480), bg)
            draw = ImageDraw.Draw(img, "RGBA")
            if "xray" in name:
                for x in [250, 390]:
                    draw.ellipse((x - 95, 82, x + 95, 375), fill=(215, 225, 230, 76), outline=(238, 246, 248, 140), width=3)
                draw.rectangle((306, 70, 334, 390), fill=(238, 242, 245, 70))
                draw.arc((220, 310, 420, 470), 195, 345, fill=(238, 242, 245, 120), width=4)
            elif "ct" in name:
                draw.ellipse((135, 55, 505, 425), fill=(20, 20, 24, 255), outline=(190, 190, 190, 220), width=5)
                draw.ellipse((205, 115, 445, 365), fill=(104, 104, 110, 255), outline=(220, 220, 220, 120), width=2)
                draw.ellipse((280, 176, 360, 258), fill=(42, 42, 47, 255))
                draw.line((320, 115, 320, 365), fill=(230, 230, 230, 50), width=2)
            elif "retina" in name:
                draw.ellipse((90, 10, 550, 470), fill=(125, 48, 32, 255))
                draw.ellipse((385, 185, 455, 255), fill=(235, 188, 86, 235))
                for y in range(160, 330, 28):
                    draw.line((420, 220, 180, y), fill=(240, 180, 110, 130), width=3)
            else:
                for i in range(90):
                    x = (i * 47) % 640
                    y = (i * 83) % 480
                    draw.ellipse((x, y, x + 54, y + 28), fill=(145, 67, 124, 58), outline=(88, 28, 82, 68))
                img = img.filter(ImageFilter.GaussianBlur(0.6))
            draw.rounded_rectangle((24, 22, 160, 64), radius=8, fill=(0, 0, 0, 145))
            draw.text((42, 35), mark, fill=(255, 255, 255))
            img.save(path)
    

def _ensure_samples() -> List[Tuple[str, str]]:
    _ensure_default_samples()
    meta = _sample_meta()
    out: List[Tuple[str, str]] = []
    used_paths = set()

    for label, values in meta.items():
        path = _find_sample_file(label, values)
        if path is None:
            continue
        out.append((str(path), label))
        used_paths.add(path.resolve())

    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if resolved in used_paths:
            continue
        out.append((str(path), _label_from_path(path)))
        used_paths.add(resolved)

    return out


def _load_model() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32
    device = "cuda" if has_cuda else "cpu"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config.text_config.attn_implementation = "eager"
        config.text_config.use_cache = True
    config.attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if has_cuda:
        model.to(device)

    _BUNDLE = ModelBundle(tokenizer=tokenizer, image_processor=image_processor, model=model, device=device, dtype=dtype)
    return _BUNDLE


def _build_prompt(task: str, body_system: str, user_prompt: str) -> str:
    base = TASK_PROMPTS.get(task, TASK_PROMPTS["VQA"])
    custom = (user_prompt or "").strip()
    content = (
        f"<image>\nClinical area: {body_system}.\nTask: {base}\n"
        "Answer once, concisely. Do not repeat the question."
    )
    if custom:
        content += f"\nUser question: {custom}"
    messages = [{"role": "user", "content": content}]
    bundle = _load_model()
    return bundle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _clean_generation(text: str, user_prompt: str) -> str:
    text = (text or "").strip()
    for marker in ("<|im_end|>", "<|im_start|>", "User:", "用户：", "Assistant:", "assistant"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    prompt = (user_prompt or "").strip()
    if prompt and prompt in text:
        text = text.split(prompt, 1)[0].strip()

    lines = []
    seen = set()
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean in seen:
            break
        seen.add(clean)
        lines.append(clean)
    return "\n".join(lines).strip()


def run_inference(
    image: Optional[Image.Image],
    task: str,
    body_system: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, str]:
    if image is None:
        return "Upload or select an image first.", "idle"
    start = time.time()
    bundle = _load_model()
    prompt = _build_prompt(task, body_system, user_prompt)
    image = image.convert("RGB")
    tok = bundle.tokenizer(prompt, return_tensors="pt")
    pix = bundle.image_processor(images=image, return_tensors="pt")
    inputs = {
        "input_ids": tok["input_ids"].to(bundle.device),
        "attention_mask": tok["attention_mask"].to(bundle.device),
        "pixel_values": pix["pixel_values"].to(bundle.device, dtype=bundle.dtype),
    }
    eos_ids = [bundle.tokenizer.eos_token_id]
    im_end_id = bundle.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            top_p=0.9,
            repetition_penalty=1.12,
            no_repeat_ngram_size=5,
            eos_token_id=eos_ids,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    text = bundle.tokenizer.decode(new_tokens[0], skip_special_tokens=False).strip()
    text = _clean_generation(text, user_prompt)
    status = f"{bundle.device.upper()} | {time.time() - start:.1f}s | {MODEL_PATH}"
    return text or "(empty response)", status


def load_example(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_named_example(sample_name: str) -> Image.Image:
    samples = _ensure_samples()
    by_name = {label: path for path, label in samples}
    return load_example(by_name.get(sample_name, samples[0][0]))


def _sample_defaults(label: str) -> Tuple[str, str, str]:
    meta = _sample_meta().get(label, {})
    return (
        meta.get("question", "Describe this medical image concisely."),
        meta.get("body_system", "Chest / Lung"),
        meta.get("task", "Report"),
    )


def select_sample(evt: gr.SelectData) -> Tuple[Image.Image, str, str, str]:
    samples = _ensure_samples()
    index = 0 if evt is None or evt.index is None else int(evt.index)
    index = max(0, min(index, len(samples) - 1))
    path, label = samples[index]
    question, body_system, task = _sample_defaults(label)
    return load_example(path), question, body_system, task


def _hero_data_uri() -> str:
    if not HERO_IMAGE.exists():
        return ""
    payload = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _image_data_uri(path: str) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_demo() -> gr.Blocks:
    samples = _ensure_samples()
    initial_question, initial_body, initial_task = _sample_defaults(samples[0][1])
    hero_uri = _hero_data_uri()
    css = """
    :root { --radius-lg: 8px; }
    .gradio-container {{ max-width: 1480px !important; background: #f6f8fb; }}
    .top-band {
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(6, 18, 33, .96) 0%, rgba(8, 30, 45, .86) 43%, rgba(6, 18, 33, .36) 100%),
        url('__HERO_URI__') center right / cover no-repeat;
      color: white;
      padding: 30px 30px 28px 30px;
      border-radius: 12px;
      margin: 8px 0 16px 0;
      min-height: 230px;
      box-shadow: 0 18px 48px rgba(15, 23, 42, .18);
    }
    .eyebrow {
      color: #7dd3fc;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .top-band h1 {
      color: #ffffff;
      font-size: 38px;
      line-height: 1.05;
      margin: 0 0 12px 0;
      letter-spacing: 0;
      max-width: 720px;
    }
    .top-band p {
      max-width: 760px;
      margin: 0;
      font-size: 15px;
      line-height: 1.55;
      color: #eef8fb;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(128px, 1fr));
      gap: 10px;
      max-width: 760px;
      margin-top: 18px;
    }
    .metric {
      border: 1px solid rgba(255,255,255,.22);
      padding: 12px;
      border-radius: 8px;
      background: rgba(255,255,255,.10);
      backdrop-filter: blur(8px);
    }
    .metric b { display:block; color:#ffffff; font-size: 20px; line-height: 1.1; }
    .metric span { display:block; color:#dce8ec; font-size:12px; margin-top: 5px; }
    .resource-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
    .resource-link {
      display:inline-flex;
      align-items:center;
      gap:7px;
      color:#f8fafc !important;
      text-decoration:none !important;
      border:1px solid rgba(255,255,255,.25);
      background: rgba(15, 23, 42, .45);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }
    .resource-link:hover { background: rgba(14, 116, 144, .65); }
    .main-panel, .side-panel {
      border: 1px solid #dbe3ea;
      background: #ffffff;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
    }
    .sample-gallery { width: calc(100% - 16px); margin: -4px 8px 12px 8px; }
    .sample-gallery .grid-wrap { border: 0 !important; background: transparent !important; }
    .sample-gallery img { border-radius: 7px !important; }
    .sample-gallery .thumbnail-item { border-radius: 8px !important; }
    button.primary { border-radius: 8px !important; }
    textarea, input, select { border-radius: 8px !important; }
    """.replace("__HERO_URI__", hero_uri).replace("{{", "{").replace("}}", "}")
    with gr.Blocks(css=css, title="GMAI-VL Medical Multimodal Console") as demo:
        gr.HTML(
            """
            <section class="top-band">
              <div class="eyebrow">General Medical Vision-Language Model</div>
              <h1>GMAI-VL Multimodal Medical Console</h1>
              <p>Run a single local 7B vision-language model on medical images and natural-language instructions for image-grounded QA, report-style impressions, localization, differential reasoning, and safety triage.</p>
              <div class="metric-grid">
                <div class="metric"><b>7B</b><span>local model scale</span></div>
                <div class="metric"><b>Image + Text</b><span>multimodal input</span></div>
                <div class="metric"><b>Multi-part</b><span>body-system coverage</span></div>
                <div class="metric"><b>Multi-task</b><span>QA, report, localization</span></div>
              </div>
              <div class="resource-row">
                <a class="resource-link" href="https://github.com/uni-medical/GMAI-VL" target="_blank">GitHub · uni-medical/GMAI-VL</a>
                <a class="resource-link" href="https://huggingface.co/General-Medical-AI" target="_blank">Hugging Face · General-Medical-AI</a>
                <a class="resource-link" href="https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M" target="_blank">Dataset card</a>
                <a class="resource-link" href="https://arxiv.org/abs/2411.14522" target="_blank">arXiv 2411.14522</a>
              </div>
            </section>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["main-panel"]):
                image = gr.Image(value=load_example(samples[0][0]), type="pil", label="Medical image", height=430)
                gallery = gr.Gallery(
                    value=samples,
                    label="Samples",
                    columns=4,
                    rows=max(2, min(4, (len(samples) + 3) // 4)),
                    height=252,
                    object_fit="cover",
                    allow_preview=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    elem_classes=["sample-gallery"],
                )
            with gr.Column(scale=4, elem_classes=["side-panel"]):
                with gr.Row():
                    task = gr.Dropdown(list(TASK_PROMPTS), value=initial_task, label="Task")
                    body = gr.Dropdown(BODY_SYSTEMS, value=initial_body, label="Body system")
                prompt = gr.Textbox(
                    label="Question",
                    value=initial_question,
                    lines=4,
                )
                with gr.Row():
                    max_tokens = gr.Slider(64, 768, value=160, step=32, label="Max tokens")
                    temperature = gr.Slider(0, 1.0, value=0.2, step=0.05, label="Temperature")
                run = gr.Button("Run GMAI-VL", variant="primary")
                status = gr.Textbox(label="Runtime", value="model loads on first run", interactive=False)
                output = gr.Textbox(label="Model answer", lines=12)
                gallery.select(select_sample, None, [image, prompt, body, task])
                run.click(run_inference, [image, task, body, prompt, max_tokens, temperature], [output, status])

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10083"))
    build_demo().queue(max_size=16).launch(server_name="0.0.0.0", server_port=port, share=False)
