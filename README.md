# GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI

<p align="center">
    <a href="https://arxiv.org/abs/2411.14522"><img src="https://img.shields.io/badge/arXiv-2411.14522-b31b1b.svg?style=flat-square" alt="arXiv"></a>
    <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39485/43446"><img src="https://img.shields.io/badge/AAAI-2026-brightgreen?style=flat-square" alt="AAAI"></a>
    <a href="https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow?style=flat-square" alt="Hugging Face"></a>
</p>

Welcome to the official code repository for **GMAI-VL & GMAI-VL-5.5M** (AAAI 2026). This repository provides the training code needed for reproducing the results and furthering research in medical AI through vision-language models.

- **GMAI-VL**: A state-of-the-art general medical vision-language model that achieves **88.48%** on OmniMedVQA with only 7B parameters, surpassing models with 34B+ parameters.
- **GMAI-VL-5.5M**: The largest open-source medical multimodal dataset with **5.5 million** QA pairs from **219** professional medical data sources, covering **13** imaging modalities and **18** clinical departments. Available on [Hugging Face](https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M).


## 📊 GMAI-VL-5.5M Dataset

GMAI-VL-5.5M is systematically constructed from **219 professional medical data sources** using an **Annotation-Guided Data Generation** pipeline — all text is grounded in expert annotations, not hallucinated.

### Dataset Subsets

| Subset | Size | Type | Description |
|:---|:---|:---|:---|
| GMAI-MM-Caption | 1.7M | Multimodal | High-quality medical image captions |
| GMAI-MM-Percept | 1.3M | Multimodal | Medical image classification & segmentation labels |
| GMAI-MM-Instrunct | 0.9M | Multimodal | Medical image analysis instruction QA |
| GMAI-Text-Single | 1.0M | Text-only | Single-turn medical text QA |
| GMAI-Text-Multi | 0.7M | Text-only | Multi-turn medical text QA |

### Comparison with Existing Datasets

| Dataset | Scale | Modalities | Languages | Traceable | Source |
|:---|:---|:---|:---|:---|:---|
| PathVQA | 32.7K | Pathology | EN | ✗ | Textbooks |
| MIMIC-CXR | 227K | X-Ray | EN | ✓ | Hospital |
| PMC-OA | 1.65M | Multi | EN | ✗ | PubMed |
| PubMedVision | 1.29M | Multi | EN&CN | ✗ | PubMed |
| **GMAI-VL-5.5M** | **5.5M** | **Multi (13)** | **EN&CN** | **✓** | **219 medical datasets** |

👉 Download: [🤗 Hugging Face](https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M)

## 🏥 GMAI-VL Model

GMAI-VL is built on the LLaVA architecture with **InternLM2.5-7B** (LLM) + **CLIP Vision Encoder** + **MLP Projector**, trained using a three-stage progressive strategy:

| Stage | Strategy | Trainable Components | Learning Rate |
|:---|:---|:---|:---|
| Stage I | Shallow Alignment | Projector only | 1e-3 |
| Stage II | Deep Alignment | Projector + Vision Encoder | 1e-4 |
| Stage III | Instruction Tuning | Full model | 1e-5 |

### Benchmark Results

| Model | Params | OmniMedVQA | GMAI-MMBench | MMMU H&M | VQA-RAD |
|:---|:---|:---|:---|:---|:---|
| InternVL2 | 40B | 78.70 | — | — | — |
| HuatuoGPT-Vision | 34B | 73.23 | — | 50.3 | — |
| medgemma | 4B | 81.92 | — | 43.3 | — |
| **GMAI-VL** | **7B** | **88.48** | **62.43** | **51.3** | **66.3** |

> With only **7B** parameters, GMAI-VL outperforms models with 34B+ parameters on multiple benchmarks, demonstrating the value of **high-quality data + progressive training**.

## 🛠️ Model Training Instructions

### 1. Installation

To set up the environment for training, please follow the instructions in the [xtuner repository](https://github.com/InternLM/xtuner). Ensure that all dependencies are correctly installed.
```bash
git clone https://github.com/uni-medical/GMAI-VL
cd GMAI-VL
pip install -e .
```
### 2. Data Preparation

Prepare your training data in the format shown in examples/example_data.json. To support multiple JSON datasets with different sampling ratios, use the format defined in examples/example_list.json.

Example structure for example_list.json:
```
{
    "FILE1": {
        "image_dir": "",
        "annotations": "examples/example_data.json",
        "sample_ratio": 1.0,
        "length": 38276,
        "data_augment": true
    },
    ...
}
```
### 3. Training

Here is a reference script to start training:

```bash
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34220


export NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
export PORT=${MASTER_PORT}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export ADDR=${MASTER_ADDR}
export HF_HOME=~/.cache
export USE_TRITON_KERNEL=1

torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    tools/llava/llava_sft.py \
    --freeze-llm \
    --freeze-vit \
    --llava $base_model_path/ \
    --chat-template internlm2 \
    --datasets examples/examples_list.json \
    --num-workers 6 \
    --mirco-batch-size $MIRCO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 1e-5 \
    --wd 0 \
    --dset-pack-level soft \
    --shard-strategy 'zero2' \
    --group-by-length \
    --resume \
    --max-length 2048 \
    --checkpoint-interval 500 \
    --log-interval 10 \
    --work-dir $work_dir/ \
    --dset-cache-dir $work_dir/cache/ \
    --dset-from-cache
  ```
💡 Note: Our training follows a multi-stage strategy. At each stage, different components (e.g., LLM or ViT) may be frozen or fine-tuned. Please adjust flags such as `--freeze-llm`, `--freeze-vit`, and learning rates accordingly, as described in the paper.

## 📈 Evaluation

For evaluation, we use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). GMAI-VL has been evaluated on **7 mainstream medical multimodal benchmarks**: OmniMedVQA, GMAI-MMBench, MMMU (Health & Medicine), VQA-RAD, SLAKE, PMC-VQA, and PathVQA. See the [paper](https://arxiv.org/abs/2411.14522) for full results.

## 📦 Open-Source Resources

| Resource | Link |
|:---|:---|
| **Paper (AAAI 2026)** | [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/39485/43446) |
| **Paper (arXiv)** | [arXiv:2411.14522](https://arxiv.org/abs/2411.14522) |
| **GMAI-VL-5.5M Dataset** | [🤗 Hugging Face](https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M) |
| **Training Code** | This repository |
| **Training Framework** | [XTuner](https://github.com/InternLM/xtuner) |
| **Evaluation Toolkit** | [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) |

## 🔗 Stay Connected

For inquiries, collaboration opportunities, or access requests, feel free to reach out via email or open a GitHub issue.

Thank you for your interest and support!

## 🙏 Acknowledgements

We would like to express our sincere gratitude to the open-source community. Our work is built upon the excellent contributions of the following toolkits:

- **[XTuner](https://github.com/InternLM/xtuner)**: An efficient, flexible, and full-featured toolkit for fine-tuning large language and vision-language models. It served as the core training framework for GMAI-VL.
- **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**: A comprehensive evaluation toolkit for large vision-language models (LVLMs). It enabled us to conduct rigorous and standardized evaluations across multiple benchmarks.

We deeply appreciate the efforts of the developers and contributors behind these projects.

## 📄 Citation

If you find our work helpful in your research, please consider citing us:
```bibtex
@inproceedings{li2026gmai,
  title={Gmai-vl \& gmai-vl-5.5 m: A large vision-language model and a comprehensive multimodal dataset towards general medical ai},
  author={Li, Tianbin and Su, Yanzhou and Li, Wei and Fu, Bin and Chen, Zhe and Huang, Ziyan and Wang, Guoan and Ma, Chenglong and Chen, Ying and Hu, Ming and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={23177--23185},
  year={2026}
}
