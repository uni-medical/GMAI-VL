# GMAI-VL Visualization Demo

This directory contains a standalone Gradio visualization interface for GMAI-VL. It loads a local GMAI-VL model checkpoint and provides a screenshot-friendly multimodal medical demo with image upload, sample images, body-system selection, task selection, and model answers.

Official resources:

- GitHub: https://github.com/uni-medical/GMAI-VL
- Hugging Face organization: https://huggingface.co/General-Medical-AI
- Dataset card: https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M
- Paper: https://arxiv.org/abs/2411.14522

## 1. Prepare Files

Clone or copy this visualization directory to the target machine:

```bash
cd gmai_vl_demo
```

Download or copy the GMAI-VL model weights to a local directory. The expected checkpoint directory should contain files such as `config.json`, tokenizer files, processor files, and model weight shards.

Recommended layout:

```text
gmai_vl_demo/
├── app.py
├── requirements-gmai-vl-demo.txt
├── run_demo.sh
├── launch_spot_demo.sh
├── sample_assets/
└── model_weight/
    ├── config.json
    ├── tokenizer.model
    └── ...
```

## 2. Create Environment

Python 3.10 is recommended.

```bash
conda create -n gmai-vl-demo python=3.10 -y
conda activate gmai-vl-demo
```

Install PyTorch for the CUDA version available on your machine, then install the demo requirements:

```bash
cd gmai_vl_demo
pip install -r requirements-gmai-vl-demo.txt
```

If the model tries to use FlashAttention and the environment does not have it, this demo already forces eager attention in `app.py`, so FlashAttention is not required for the visualization.

## 3. Configure Model Path

Set the model checkpoint path before launching:

```bash
export GMAI_VL_MODEL_PATH=./model_weight
```

If your checkpoint is stored elsewhere, point the variable to that directory:

```bash
export GMAI_VL_MODEL_PATH=../checkpoints/model_weight
```

## 4. Run With Slurm GPU

On this cluster, GPU jobs should be launched through `srun`.

Default port is `10083`:

```bash
cd gmai_vl_demo
bash launch_spot_demo.sh
```

This script requests one GPU from the `Medisco` partition and uses spot quota by default:

```bash
srun -p Medisco --quotatype=spot --gres=gpu:1 --cpus-per-task=12 ...
```

Open the page after the job is running:

```text
http://<allocated-node-ip>:10083
```

Check job status:

```bash
squeue -u $USER
```

View logs:

```bash
tail -f logs/gmai_vl_demo_10083.log
```

## 5. Change Port

Set `PORT` when launching:

```bash
PORT=18080 bash launch_spot_demo.sh
```

Then open:

```text
http://<allocated-node-ip>:18080
```

## 6. Run Without Slurm

Use this only when you already have a visible GPU in the current shell, or when you only want to preview the UI.

```bash
cd gmai_vl_demo
PORT=10083 bash run_demo.sh
```

Open:

```text
http://<host-ip>:10083
```

## 7. Customize Samples

Sample images and their default questions are stored in:

```bash
sample_assets/
sample_assets/sample_prompts.json
```

Edit `sample_prompts.json` to bind each sample image to a specific task, body system, and Chinese or English question.

Example entry:

```json
{
  "Retina": {
    "file": "retina.png",
    "task": "VQA",
    "body_system": "Eye / Retina",
    "question": "Describe this fundus image."
  }
}
```

The gallery is loaded dynamically from `sample_prompts.json` first, then from any extra image files in `sample_assets/`.
