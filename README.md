# GenAI Projects

This repository contains a series of GenAI implementations developed as part of an advanced GenAI class. The projects cover a range of cutting-edge techniques including transformer enhancements, diffusion-based image generation, parameter-efficient fine-tuning, and text-guided image editing.

## Projects Overview

- **GPT with RoPE & GQA:** Upgraded a vanilla GPT model by integrating Rotary Position Embeddings (RoPE) and Grouped-Query Attention (GQA) to build a custom LLAMA-2 style model using the complete works of Shakespeare.
- **Diffusion-Based Image Generation:** Implemented a Denoising Diffusion Probabilistic Model (DDPM) using the Animal Faces-HQ (AFHQ) dataset (cat images) to explore state-of-the-art techniques for image denoising and generation.
- **LoRA-Enabled GPT2 Fine-Tuning:** Developed a parameter-efficient fine-tuning method using Low Rank Adaptation (LoRA) on a pre-trained GPT2 model applied to the Rotten Tomatoes sentiment analysis dataset.
- **Prompt-to-Prompt Image Editing:** Built a text-driven image editing pipeline leveraging cross-attention maps in a diffusion model framework to modify specific scene elements while preserving overall image structure.

## Requirements

- Python 3.7+
- PyTorch, TensorFlow (where applicable), and other libraries as specified in individual `requirements.txt` files for each project.
- Additional dependencies may include HuggingFace Diffusers, Wandb, and related ML/AI toolkits.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages (for each project, navigate to the project directory and install):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- **Project 1 (GPT with RoPE & GQA):**
  ```bash
  python chargpt.py --config config.yaml
