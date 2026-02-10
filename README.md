# Intro to Generative AI: Diffusion Models, LoRA & CLIP

An introductory project for learning **personalized image generation** using:
- 🎨 **Stable Diffusion** (pre-trained text-to-image model)
- ⚡ **LoRA** (Low-Rank Adaptation for efficient personalization)
- 🔍 **CLIP** (Vision-Language model for prompt-image alignment)

**Designed for ML/DL practitioners new to generative AI**, this project provides hands-on experience with modern GenAI techniques through three progressive Jupyter notebooks.

---

## 📚 Table of Contents
- [What You'll Learn](#what-youll-learn)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Conceptual Overview](#conceptual-overview)
- [Notebooks Walkthrough](#notebooks-walkthrough)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Further Learning](#further-learning)

---

## 🎯 What You'll Learn

1. **Stable Diffusion Basics** - How diffusion models generate images from text
2. **LoRA Fine-tuning** - Lightweight personalization with <1% of original parameters
3. **CLIP Reranking** - Semantic evaluation and selection of generated images

---

## 📋 Prerequisites

- **Machine Learning Fundamentals**: Understanding of neural networks, backpropagation, loss functions
- **Deep Learning Basics**: Familiarity with PyTorch, CNNs, and training loops
- **Python Skills**: Comfortable with Python, NumPy, and Jupyter notebooks
- **No GenAI Experience Required!** This project is designed for GenAI beginners

---

## 📁 Project Structure

```
intro-genai-diffusion/
├── notebooks/
│   ├── 01_inference_stable_diffusion.ipynb   # Basic SD inference
│   ├── 02_lora_finetuning.ipynb              # LoRA personalization
│   └── 03_clip_reranking.ipynb               # CLIP-based selection
├── data/
│   └── sample_images/                        # Training images (auto-downloaded)
├── outputs/                                  # Generated images
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## 🚀 Setup Instructions

### Google Colab (Recommended)

1. **Upload this project to Google Drive**
2. **Open each notebook in Google Colab**:
   - Right-click notebook → "Open with" → "Google Colaboratory"
3. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator: **T4 GPU**
4. **Install dependencies** (first cell in each notebook):
   ```python
   !pip install diffusers transformers accelerate peft torch torchvision
   ```

### Local Setup (Requires CUDA GPU)

```bash
# Clone or download this repository
cd intro-genai-diffusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 🧠 Conceptual Overview

### What are Diffusion Models?

Diffusion models learn to **reverse a noising process**:

1. **Forward Process** (Training): Gradually add noise to images until pure random noise
2. **Reverse Process** (Inference): Learn to remove noise step-by-step, guided by text

**Stable Diffusion** uses a **latent diffusion** approach:
- Operates in a compressed latent space (not pixel space)
- Much faster and more memory-efficient
- Uses a **text encoder** (CLIP) to condition generation on prompts

**Key Intuition**: The model learns "what a slightly less noisy version looks like" and iteratively applies this to generate images from pure noise.

---

### Why LoRA? Understanding Parameter-Efficient Fine-Tuning

**Problem**: Full fine-tuning of Stable Diffusion:
- 🔴 ~900M parameters to update
- 🔴 Requires days of training on expensive GPUs
- 🔴 Huge storage (multiple GB per personalized model)

**LoRA Solution**: Freeze original weights, add small trainable matrices

**Mathematical Concept**:
```
Original: Y = WX  (W is frozen, e.g., 1024×1024)
LoRA:     Y = WX + BAX  (B: 1024×8, A: 8×1024, only ~16K params!)
```

**Benefits**:
- ⚡ **0.5-1% of parameters** to train (~5-10MB vs 2GB)
- ⚡ **Minutes instead of days**
- ⚡ **Easy to swap** different LoRA adapters on same base model

**Why Attention Layers?**
LoRA targets **cross-attention layers** because they control:
- Text-image alignment (which words map to which image regions)
- Semantic understanding of concepts
- Most expressive for learning new visual styles/subjects

---

### How CLIP Enables Vision-Language Understanding

**CLIP** (Contrastive Language-Image Pre-training) learns a **joint embedding space**:
- Text and images are mapped to the same vector space
- Semantically similar text-image pairs have high cosine similarity

**Training Process**:
- Trained on 400M image-caption pairs
- Learns to match images with correct captions (and reject wrong ones)

**For Diffusion Models**:
- **Text Encoder**: Converts prompts → embeddings that guide generation
- **Image Evaluation**: Measures how well generated images match prompts
- **Reranking**: Select best image from multiple generations

---

## 📓 Notebooks Walkthrough

### Notebook 1: Stable Diffusion Inference
**Goal**: Generate images from text prompts using pre-trained SD 1.5

**What You'll Do**:
- Load a pre-trained diffusion pipeline
- Generate images with different prompts
- Experiment with generation parameters (`guidance_scale`, `num_inference_steps`)
- Understand the denoising process

**Key Takeaway**: How text-to-image generation works under the hood

**Runtime**: ~5 minutes

---

### Notebook 2: LoRA Fine-tuning
**Goal**: Personalize SD on a small dataset (10-20 images)

**What You'll Do**:
- Download sample training images (automated)
- Configure LoRA for attention layers
- Train for ~100-200 steps
- Generate personalized images with the adapted model

**Key Concepts**:
- Which layers to adapt and why
- LoRA rank (`r`) and its impact on capacity
- Efficient training with `accelerate`

**Key Takeaway**: How to personalize large models with minimal resources

**Runtime**: ~10 minutes (including training)

---

### Notebook 3: CLIP Reranking
**Goal**: Use CLIP to evaluate and select the best generated images

**What You'll Do**:
- Load CLIP vision-language model
- Generate multiple images from the same prompt
- Compute prompt-image similarity scores
- Rank and visualize results

**Key Concepts**:
- Joint embedding spaces
- Cosine similarity for semantic alignment
- Quality control in generation pipelines

**Key Takeaway**: How to evaluate generation quality beyond human inspection

**Runtime**: ~5 minutes

---

## 💻 Hardware Requirements

### Google Colab Free Tier (Recommended)
- ✅ **T4 GPU** (15GB VRAM) - sufficient for all notebooks
- ✅ **Runtime**: Each notebook runs in <15 minutes
- ⚠️ **Limitation**: Session timeout after inactivity

### Local Machine (Alternative)
- **Minimum**: NVIDIA GPU with 8GB VRAM (GTX 1070, RTX 2060)
- **Recommended**: 12GB+ VRAM (RTX 3060, 4060)
- **RAM**: 16GB system RAM
- **Storage**: ~10GB for models and outputs

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` or kernel crash

**Solutions**:
```python
# Use float16 precision
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # Half precision
)

# Enable memory-efficient attention
pipe.enable_attention_slicing()

# Reduce batch size for LoRA training
train_batch_size = 1
```

### Slow Generation

**Issue**: Each image takes >1 minute

**Solutions**:
- Reduce `num_inference_steps` (try 25-30 instead of 50)
- Ensure GPU is enabled in Colab
- Use DPM-Solver++ scheduler for faster sampling

### CLIP Scores Don't Match Visual Quality

**Why**: CLIP measures semantic alignment, not aesthetic quality

**Solution**: Combine CLIP with other metrics (FID, aesthetic predictors) for production

---

## 📖 Further Learning

### Deep Dives
- **Diffusion Models**: [DDPM Paper](https://arxiv.org/abs/2006.11239), [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- **LoRA**: [LoRA Paper](https://arxiv.org/abs/2106.09685)
- **CLIP**: [CLIP Paper](https://arxiv.org/abs/2103.00020)

### Advanced Topics
- **ControlNet**: Add spatial control (pose, depth, edges)
- **DreamBooth**: Alternative personalization method
- **Textual Inversion**: Learn new tokens for concepts
- **Img2Img**: Transform existing images with prompts

### Community Resources
- [Hugging Face Diffusers Docs](https://huggingface.co/docs/diffusers)
- [Civitai](https://civitai.com/) - Community LoRA models
- [Stable Diffusion Subreddit](https://www.reddit.com/r/StableDiffusion/)

---

## 🤝 Contributing & Feedback

This is an educational project. Feel free to:
- Open issues for bugs or unclear explanations
- Suggest improvements to make concepts more beginner-friendly
- Share your generated images!

---

## 📜 License

MIT License - Free to use for learning and educational purposes.

---

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion
- **Hugging Face** for the Diffusers library
- **OpenAI** for CLIP
- **Microsoft** for LoRA (originally for language models)

---

**Happy Generating! 🎨✨**
