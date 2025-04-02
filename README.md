# GMAI-VL & GMAI-VL-5.5M: A General Medical Vision-Language Model and Multimodal Dataset

Welcome to the GMAI-VL code repository, which accompanies the paper "GMAI-VL & GMAI-VL-5.5M: A General Medical Vision-Language Model and Multimodal Dataset." This repository provides the resources needed for reproducing the results and furthering research in medical AI through vision-language models.
This repository includes:

- **GMAI-VL**: A state-of-the-art general medical vision-language model.
- **GMAI-VL-5.5M**: A comprehensive multimodal medical dataset containing 5.5 million images and associated text, designed to support a wide range of medical AI research.


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
💡 Note: Our training follows a multi-stage strategy. At each stage, different components (e.g., LLM or ViT) may be frozen or fine-tuned. Please adjust flags such as --freeze-llm, --freeze-vit, and learning rates accordingly, as described in the paper.
## 📊 Evaluation
For evaluation, please use the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). It provides comprehensive tools for evaluating vision-language models on various tasks.

## 📅 Release Timeline

- **2025-04-02**: The training code and model weight for the GMAI-VL model has been officially released! 🎉  
This update includes detailed instructions for model training, dataset preparation, and evaluation.
- **2024-11-21**: The paper was officially released on [arXiv](https://arxiv.org/abs/2411.14522)!
- **Coming Soon**: dataset will be released. Please watch this repository for updates. We are committed to making these resources available as soon as possible. Please watch this repository or check back regularly for updates.

## 🔗 Stay Connected

For inquiries, collaboration opportunities, or access requests, feel free to reach out via email or open a GitHub issue.

Thank you for your interest and support!
## 📄 Paper and Citation

Our paper has been published on [arXiv](https://arxiv.org/abs/2411.14522). If you use our work in your research, please consider citing us:

### BibTeX Citation
```bibtex
@article{li2024gmai,
      title={GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI},
      author={Tianbin Li, Yanzhou Su, Wei Li, Bin Fu, Zhe Chen, Ziyan Huang, Guoan Wang, Chenglong Ma, Ying Chen, Ming Hu, Yanjun Li, Pengcheng Chen, Xiaowei Hu, Zhongying Deng, Yuanfeng Ji, Jin Ye, Yu Qiao, Junjun He},
  journal={arXiv preprint arXiv:2411.14522},
  year={2024}
}