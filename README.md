# FlowVideoSR
Efficient video super-resolution framework via flow-based T2V model

<!-- ### TODO list sorted by priority

<details>
<summary>View more</summary>

- [x] Training Video-VAE and adapt our model to new VAE.
- [x] Scaling model parameters and dataset size.
- [x] Incoporate a better scheduler (rectified flow).
- [x] Evaluation pipeline.
- [x] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity, etc.). See [the dataset](/docs/datasets.md) for more information
- [x] Support image and video conditioning.
- [x] Support variable aspect ratios, resolutions, durations.

</details> -->


## Installation

### Install VRT
We can optionally use VRT as the first-stage restorer for severely degraded video. The pre-trained weight can be downloaded [here](https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth). 

```bash
# download code of VRT
git clone https://github.com/JingyunLiang/VRT

# first-stage restoration
python main_test_vrt.py --task 001_VRT_videosr_bi_REDS_6frames --folder_lq YOUR_LQ_PATH --folder_gt YOUR_GT_PATH --tile 40 128 128 --tile_overlap 2 20 20 --save_result
```

### Install OpenSora Environment from Source

For CUDA 12.1, you can install the dependencies with the following commands. Otherwise, please refer to [Installation Documentation](docs/installation.md) for more instructions on different cuda version, and additional dependency for data preprocessing, VAE, and model evaluation.

```bash
# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.9
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# install torch, torchvision and xformers
pip install -r requirements/requirements-cu121.txt

# the default installation is for inference only
pip install -v . # for development mode, `pip install -v -e .`
```

(Optional, recommended for fast speed, especially for training) To enable `layernorm_kernel` and `flash_attn`, you need to install `apex` and `flash-attn` with the following commands.

```bash
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex
# set enable_layernorm_kernel=False in config to disable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

<!-- ### Use Docker

Run the following command to build a docker image from Dockerfile provided.

```bash
docker build -t opensora .
```

Run the following command to start the docker container in interactive mode.

```bash
docker run -ti --gpus all -v .:/workspace/Open-Sora opensora
``` -->

## Model Weights

### CogVideoX Model Weight
See **[this link](https://huggingface.co/THUDM/CogVideoX-2b)** for more infomation. Weight will be automatically downloaded when you run the inference script.

Please download our **[pre-trained model](https://cloud.tsinghua.edu.cn/d/a8f2a17b5f2c4b378244/)** for inference and finetuning. 
### Getting Started


## Inference

### FlowVideoSR Command Line Inference

The basic command line inference is as follows:

```bash
# enhance a video
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2  scripts/inference_sr_tiled_chunked_lora_controlnext.py ./configs/CogVideoX_sr/inference/sample_apple_lora_controlnext.py
```

To enable sequence parallelism, you need to use `torchrun` to run the inference script. The following command will run the inference with 2 GPUs.


### CogVideo Training

<details>
<summary>View more</summary>

Once you prepare the data in a `csv` file, run the following commands to launch training on a single node.

```bash
# one node
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 ./scripts/train_sr_woema_lora_controlnext.py ./configs/CogVideoX_sr/train/sr_YouHQ_video_compression_lora_controlnext.py
```

</details>



