# FlowVideoSR
Efficient video super-resolution framework via flow-based T2V model

We design and implement **Open-Sora**, an initiative dedicated to **efficiently** producing high-quality video. We hope to make the model,
tools and all details accessible to all. By embracing **open-source** principles,
Open-Sora not only democratizes access to advanced video generation techniques, but also offers a
streamlined and user-friendly platform that simplifies the complexities of video generation.
With Open-Sora, our goal is to foster innovation, creativity, and inclusivity within the field of content creation.

[[中文文档](/docs/zh_CN/README.md)] [[潞晨云](https://cloud.luchentech.com/)|[OpenSora镜像](https://cloud.luchentech.com/doc/docs/image/open-sora/)|[视频教程](https://www.bilibili.com/video/BV1ow4m1e7PX/?vd_source=c6b752764cd36ff0e535a768e35d98d2)]

## 📰 News

- **[2024.06.17]** 🔥 We released **Open-Sora 1.2**, which includes **3D-VAE**, **rectified flow**, and **score condition**. The video quality is greatly improved. [[checkpoints]](#open-sora-10-model-weights) [[report]](/docs/report_03.md)   [[blog]](https://hpc-ai.com/blog/open-sora-from-hpc-ai-tech-team-continues-open-source-generate-any-16-second-720p-hd-video-with-one-click-model-weights-ready-to-use)
- **[2024.04.25]** 🤗 We released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces.
- **[2024.04.25]** We released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]]() [[report]](/docs/report_02.md)
- **[2024.03.18]** We released **Open-Sora 1.0**, a fully open-source project for video generation.
  Open-Sora 1.0 supports a full pipeline of video data preprocessing, training with
  <a href="https://github.com/hpcaitech/ColossalAI"><img src="assets/readme/colossal_ai.png" width="8%" ></a>
  acceleration,
  inference, and more. Our model can produce 2s 512x512 videos with only 3 days training. [[checkpoints]](#open-sora-10-model-weights)
  [[blog]](https://hpc-ai.com/blog/open-sora-v1.0) [[report]](/docs/report_01.md)
- **[2024.03.04]** Open-Sora provides training with 46% cost reduction.
  [[blog]](https://hpc-ai.com/blog/open-sora)

## 🎥 Latest Demo

🔥 You can experience Open-Sora on our [🤗 Gradio application on Hugging Face](https://huggingface.co/spaces/hpcai-tech/open-sora). More samples and corresponding prompts are available in our [Gallery](https://hpcaitech.github.io/Open-Sora/).

| **4s 720×1280**                                                                                                                                      | **4s 720×1280**                                                                                                                                      | **4s 720×1280**                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/v1.2/sample_0013.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/7895aab6-ed23-488c-8486-091480c26327) | [<img src="assets/demo/v1.2/sample_1718.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/20f07c7b-182b-4562-bbee-f1df74c86c9a) | [<img src="assets/demo/v1.2/sample_0087.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/3d897e0d-dc21-453a-b911-b3bda838acc2) |
| [<img src="assets/demo/v1.2/sample_0052.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/644bf938-96ce-44aa-b797-b3c0b513d64c) | [<img src="assets/demo/v1.2/sample_1719.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/272d88ac-4b4a-484d-a665-8d07431671d0) | [<img src="assets/demo/v1.2/sample_0002.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ebbac621-c34e-4bb4-9543-1c34f8989764) |
| [<img src="assets/demo/v1.2/sample_0011.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/a1e3a1a3-4abd-45f5-8df2-6cced69da4ca) | [<img src="assets/demo/v1.2/sample_0004.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/d6ce9c13-28e1-4dff-9644-cc01f5f11926) | [<img src="assets/demo/v1.2/sample_0061.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/561978f8-f1b0-4f4d-ae7b-45bec9001b4a) |

<details>
<summary>OpenSora 1.1 Demo</summary>

| **2s 240×426**                                                                                                                                              | **2s 240×426**                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sample_16x240x426_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) | [<img src="assets/demo/sora_16x240x426_26.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c31ebc52-de39-4a4e-9b1e-9211d45e05b2) |
| [<img src="assets/demo/sora_16x240x426_27.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/f7ce4aaa-528f-40a8-be7a-72e61eaacbbd)  | [<img src="assets/demo/sora_16x240x426_40.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/5d58d71e-1fda-4d90-9ad3-5f2f7b75c6a9) |

| **2s 426×240**                                                                                                                                             | **4s 480×854**                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sora_16x426x240_24.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/34ecb4a0-4eef-4286-ad4c-8e3a87e5a9fd) | [<img src="assets/demo/sample_32x480x854_9.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/c1619333-25d7-42ba-a91c-18dbc1870b18) |

| **16s 320×320**                                                                                                                                        | **16s 224×448**                                                                                                                                        | **2s 426×240**                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/demo/sample_16s_320x320.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/3cab536e-9b43-4b33-8da8-a0f9cf842ff2) | [<img src="assets/demo/sample_16s_224x448.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/9fb0b9e0-c6f4-4935-b29e-4cac10b373c4) | [<img src="assets/demo/sora_16x426x240_3.gif" width="">](https://github.com/hpcaitech/Open-Sora-dev/assets/99191637/3e892ad2-9543-4049-b005-643a4c1bf3bf) |

</details>

<details>
<summary>OpenSora 1.0 Demo</summary>

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="assets/readme/sample_0.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/de1963d3-b43b-4e68-a670-bb821ebb6f80)                                 | [<img src="assets/readme/sample_1.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/13f8338f-3d42-4b71-8142-d234fbd746cc)                              | [<img src="assets/readme/sample_2.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/fa6a65a6-e32a-4d64-9a9e-eabb0ebb8c16)    |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall. |
| [<img src="assets/readme/sample_3.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/64232f84-1b36-4750-a6c0-3e610fa9aa94)                                 | [<img src="assets/readme/sample_4.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/983a1965-a374-41a7-a76b-c07941a6c1e9)                              | [<img src="assets/readme/sample_5.gif" width="">](https://github.com/hpcaitech/Open-Sora/assets/99191637/ec10c879-9767-4c31-865f-2e8d6cf11e65)    |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                   |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display,
see [here](/assets/texts/t2v_samples.txt) for full prompts.

</details>

## 🔆 New Features/Updates

- 📍 **Open-Sora 1.2** released. Model weights are available [here](#model-weights). See our **[report 1.2](/docs/report_03.md)** for more details.
- ✅ Support rectified flow scheduling.
- ✅ Support more conditioning including fps, aesthetic score, motion strength and camera motion.
- ✅ Trained our 3D-VAE for temporal dimension compression.
- 📍 **Open-Sora 1.1** released. Model weights are available [here](#model-weights). It is trained on **0s~15s, 144p to 720p, various aspect ratios** videos. See our **[report 1.1](/docs/report_02.md)** for more discussions.
- 🔧 **Data processing pipeline v1.1** is released. An automatic [processing pipeline](#data-processing) from raw videos to (text, video clip) pairs is provided, including scene cutting $\rightarrow$ filtering(aesthetic, optical flow, OCR, etc.) $\rightarrow$ captioning $\rightarrow$ managing. With this tool, you can easily build your video dataset.

<details>
<summary>View more</summary>

- ✅ Improved ST-DiT architecture includes rope positional encoding, qk norm, longer text length, etc.
- ✅ Support training with any resolution, aspect ratio, and duration (including images).
- ✅ Support image and video conditioning and video editing, and thus support animating images, connecting videos, etc.
- 📍 **Open-Sora 1.0** released. Model weights are available [here](#model-weights). With only 400K video clips and 200 H800
  days (compared with 152M samples in Stable Video Diffusion), we are able to generate 2s 512×512 videos. See our **[report 1.0](docs/report_01.md)** for more discussions.
- ✅ Three-stage training from an image diffusion model to a video diffusion model. We provide the weights for each
  stage.
- ✅ Support training acceleration including accelerated transformer, faster T5 and VAE, and sequence parallelism.
  Open-Sora improves **55%** training speed when training on 64x512x512 videos. Details locates
  at [acceleration.md](docs/acceleration.md).
- 🔧 **Data preprocessing pipeline v1.0**,
  including [downloading](tools/datasets/README.md), [video cutting](tools/scene_cut/README.md),
  and [captioning](tools/caption/README.md) tools. Our data collection plan can be found
  at [datasets.md](docs/datasets.md).
- ✅ We find VQ-VAE from [VideoGPT](https://wilson1yan.github.io/videogpt/index.html) has a low quality and thus adopt a
  better VAE from [Stability-AI](https://huggingface.co/stabilityai/sd-vae-ft-mse-original). We also find patching in
  the time dimension deteriorates the quality. See our **[report](docs/report_01.md)** for more discussions.
- ✅ We investigate different architectures including DiT, Latte, and our proposed STDiT. Our **STDiT** achieves a better
  trade-off between quality and speed. See our **[report](docs/report_01.md)** for more discussions.
- ✅ Support clip and T5 text conditioning.
- ✅ By viewing images as one-frame videos, our project supports training DiT on both images and videos (e.g., ImageNet &
  UCF101). See [commands.md](docs/commands.md) for more instructions.
- ✅ Support inference with official weights
  from [DiT](https://github.com/facebookresearch/DiT), [Latte](https://github.com/Vchitect/Latte),
  and [PixArt](https://pixart-alpha.github.io/).
- ✅ Refactor the codebase. See [structure.md](docs/structure.md) to learn the project structure and how to use the
  config files.

</details>

### TODO list sorted by priority

<details>
<summary>View more</summary>

- [x] Training Video-VAE and adapt our model to new VAE.
- [x] Scaling model parameters and dataset size.
- [x] Incoporate a better scheduler (rectified flow).
- [x] Evaluation pipeline.
- [x] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity, etc.). See [the dataset](/docs/datasets.md) for more information
- [x] Support image and video conditioning.
- [x] Support variable aspect ratios, resolutions, durations.

</details>

## Contents

- [Installation](#installation)
- [Model Weights](#model-weights)
- [Gradio Demo](#gradio-demo)
- [Inference](#inference)
- [Data Processing](#data-processing)
- [Training](#training)
- [Evaluation](#evaluation)
- [VAE Training & Evaluation](#vae-training--evaluation)
- [Contribution](#contribution)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

- Report: each version is trained from a image base seperately (not continuously trained), while a newer version will incorporate the techniques from the previous version.
  - [report 1.2](docs/report_03.md): rectified flow, 3d-VAE, score condition, evaluation, etc.
  - [report 1.1](docs/report_02.md): multi-resolution/length/aspect-ratio, image/video conditioning/editing, data preprocessing, etc.
  - [report 1.0](docs/report_01.md): architecture, captioning, etc.
  - [acceleration.md](docs/acceleration.md)
- Repo structure: [structure.md](docs/structure.md)
- Config file explanation: [config.md](docs/config.md)
- Useful commands: [commands.md](docs/commands.md)
- Data processing pipeline and dataset: [datasets.md](docs/datasets.md)
- Each data processing tool's README: [dataset conventions and management](/tools/datasets/README.md), [scene cutting](/tools/scene_cut/README.md), [scoring](/tools/scoring/README.md), [caption](/tools/caption/README.md)
- Evaluation: [eval/README.md](/eval/README.md)
- Gallery: [gallery](https://hpcaitech.github.io/Open-Sora/)

## Installation

### Install VRT
We can optionally use VRT as the first-stage restorer for severely degraded video. The pre-trained weight can be downloaded [here](https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth). 

```bash
# download code of VRT
git clone https://github.com/JingyunLiang/VRT

# first-stage restoration
python main_test_vrt.py --task 001_VRT_videosr_bi_REDS_6frames --folder_lq YOUR_LQ_PATH --folder_gt YOUR_GT_PATH --tile 40 128 128 --tile_overlap 2 20 20 --save_result
```

### Install OpenSora from Source

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

### Use Docker

Run the following command to build a docker image from Dockerfile provided.

```bash
docker build -t opensora .
```

Run the following command to start the docker container in interactive mode.

```bash
docker run -ti --gpus all -v .:/workspace/Open-Sora opensora
```

## Model Weights

### CogVideoX Model Weight
See **[this link](https://huggingface.co/THUDM/CogVideoX-2b)** for more infomation. Weight will be automatically downloaded when you run the inference script.


### Getting Started


## Inference

### Open-Sora 1.2 Command Line Inference

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



