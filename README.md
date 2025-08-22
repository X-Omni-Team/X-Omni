# 🎨 X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again

<p>
  <a href="https://x-omni-team.github.io">🏠 Project Page</a> |
  <a href="https://arxiv.org/pdf/2507.22058">📄 Paper</a> |
  <a href="https://huggingface.co/collections/X-Omni/x-omni-models-6888aadcc54baad7997d7982">🤗 Model</a> |
  <a href="https://huggingface.co/collections/X-Omni/x-omni-spaces-6888c64f38446f1efc402de7">🚀 Space</a> |
  <a href="#my-benchmark-section">📊 LongText-Bench</a> |
  <a href="https://huggingface.co/datasets/X-Omni/LongText-Bench">🤗 LongText-Bench</a>
</p>

Official inference code and LongText-Bench benchmark for our paper **X-Omni**, a unified discrete autoregressive model for both image and language modalities.

## 🔥News

- **2025.8.22**  To facilitate the fine-tuning of X-Omni, we have open-sourced the checkpoints from both the pre-training stage and the supervised fine-tuning stage. ([X-Omni-PT](https://huggingface.co/X-Omni/X-Omni-PT), [X-Omni-SFT](https://huggingface.co/X-Omni/X-Omni-SFT))

## 🌟 Highlights

- **Unified Modeling Approach**: A discrete autoregressive model handling image and language modalities.
- **Superior Instruction Following**: Exceptional capability to follow complex instructions.
- **Superior Text Rendering**: Accurately render text in multiple languages, including both English and Chinese.
- **Arbitrary resolutions**: Produces aesthetically pleasing images at arbitrary resolutions.

## 🚀 Quick Start Guide

### Installation
```bash
conda create -n xomni python==3.12
conda activate xomni
pip install -r requirements.txt
pip install flash-attn --no-build-isolation 
```
If you get trouble in installing flash_attn, please refer to the offical repository: https://github.com/Dao-AILab/flash-attention. We recommand installing from .whl file like:
```bash
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
### Inference
#### 1. Image Generation in English
```bash
PROMPT='A formal letter document with a professional tone. Create a document that includes  a section starting with "To, Mr. Edward Robertson," aligned to the left. Underneath, place the date "Date: 27th July 2025" also aligned to the left. Begin the body of the letter with "Dear Sir," indented slightly from the left margin. The first paragraph should state, "I am writing to you with intent of purchasing your property located at #765, Lincoln Street, New York." The second paragraph should read, "I want to propose a purchase price of $100,000 for your property. I am willing to pay you $20,000 as advance." The closing remarks should be, "Kindly let me know what do you think of the offer and we can make a few changes as per your requirements." followed by "Regards," and then "William Specter". Finally, add a logo with a feather graphic in the bottom right corner.'

IMG_PATH=/path/to/save/generation
FLUX_PATH=/path/to/FLUX.1-dev
python generate.py \
    --model_name_or_path X-Omni/X-Omni-En \
    --flux_model_name_or_path $FLUX_PATH \
    --prompt "$PROMPT" \
    --image-size 1152 1152 \
    --cfg-scale 1.0 \
    --min-p 0.03 \
    --seed 1234 \
    --output-path $IMG_PATH
```

#### 2. Image Generation in Chinese
```bash
PROMPT='生成一张雪中的紫禁城全景封面图，作为北京冬季旅游指南的主题。画面以近景构图展现建筑，红墙金瓦被皑皑白雪覆盖，朱红色宫墙，金黄色瓦片与洁白雪色形成强烈对比，琉璃瓦顶的积雪在阳光下折射出晶莹光泽。前景一枝腊梅花正在盛开，背景为灰蓝色冬日天空，飘落细雪，远处角楼轮廓若隐若现，增添朦胧诗意感。图片上有标题“雪落北平·穿越600年”，另有副标题“北京古建筑雪景深度游”。文字艺术感极强，与图片良好融合起来'

IMG_PATH=/path/to/save/generation
FLUX_PATH=/path/to/FLUX.1-dev
python generate.py \
    --model_path X-Omni/X-Omni-Zh \
    --flux_model_name_or_path $FLUX_PATH \
    --prompt "$PROMPT" \
    --image-size 1152 1152 \
    --cfg-scale 1.0 \
    --min-p 0.03 \
    --seed 1234 \
    --output-path $IMG_PATH
```

#### 3. Multi-modal Chat
```bash
IMG_PATH=/path/to/your/input/image
PROMPT="Describe the image in detail."
FLUX_PATH=/path/to/FLUX.1-dev
python chat.py \
    --model_name_or_path X-Omni/X-Omni-En \
    --flux_model_name_or_path $FLUX_PATH \
    --prompt $PROMPT \
    --image-path $IMG_PATH
```

<a id="my-benchmark-section"></a>
## 📊 LongText-Bench 
[huggingface dataset](https://huggingface.co/datasets/X-Omni/LongText-Bench)

### 1. Install environment for Qwen2.5-VL
```bash
pip install transformers==4.52.0
pip install qwen_vl_utils
```
### 2. Sample results
Generate images according to prompts in 'text_prompts.jsonl' and 'text_prompts_zh.jsonl' and save according to the following structure:
```
├── <SAMPLE_DIR>/
│   ├── 0000_1.png
│   ├── 0000_2.png
│   ├── 0000_3.png
│   ├── 0000_4.png
│   ├── ...
│   ├── 0199_1.png
│   ├── 0199_2.png
│   ├── 0199_3.png
│   └── 0199_4.png
```
Make sure your generation results saved in the format: {prompt_id}_{repeat_id}.png, where prompt_id is provided in the prompt file and we uniformly sample each prompt four times to calculate the final results.
### 3. Evaluation
Here we provide a distributed evaluation script with torch DDP:
```bash
cd textbench
bash eval.sh
```
Replace MODE and SAMPLE_FOLDER in this script according to your generation results in step2. Feel free to modify the related parameters according to your requirements. 
## 📖 Citation

If you find this project helpful for your research or use it in your own work, please cite our paper:
```bibtex
@article{geng2025xomni,
      author       = {Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang and Jie Jiang},
      title        = {X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again},
      journal      = {CoRR},
      volume       = {abs/2507.22058},
      year         = {2025},
}
```

---

## 📬 Contact & Feedback

For questions or feedback, please don't hesitate to reach out:

- **Yibing Wang**: wangyibing18@mails.ucas.ac.cn
- **Tencent Hunyuan X Team**

If you are interested in joining us in working on **unified multimodal models**, please feel free to contact [Xiaosong Zhang](https://zhangxiaosong18.github.io).

---

⭐️ If this repository helped your research, please star 🌟 this repo 👍!
