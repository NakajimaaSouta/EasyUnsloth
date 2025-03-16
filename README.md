# Unsloth Fine-Tuning Guide

This repository contains a simplified setup for fine-tuning language models using [Unsloth](https://github.com/unslothai/unsloth), a library that makes fine-tuning LLMs faster and more memory-efficient.

## What is Unsloth?

Unsloth is a library that optimizes the fine-tuning process for large language models. It provides:

- Up to 2x faster fine-tuning
- Lower memory usage (fits larger batch sizes)
- Support for various models (Llama, Mistral, Gemma, Phi, etc.)
- Easy integration with Hugging Face's ecosystem

## Setup Instructions

### Option 1: Using pip (Windows/Linux/macOS)

1. Create a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

2. Install the required packages:
   ```bash
   pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes
   ```

   Note: Replace `cu118` with the appropriate CUDA version for your system:
   - `cu118`: CUDA 11.8
   - `cu121`: CUDA 12.1
   - `cu122`: CUDA 12.2
   - `cpu`: No CUDA (CPU only)

### Option 2: Using conda (Recommended for complex setups)

1. Install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)

2. Create a conda environment:
   ```bash
   conda create --name unsloth_env python=3.11 -y
   conda activate unsloth_env
   ```

3. Install PyTorch with CUDA:
   ```bash
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install Unsloth:
   ```bash
   pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes
   ```

## Training a Model

You can train a model using either the Python script (`train.py`) or the Jupyter notebook (`train.ipynb`).

### Using the Python script

```bash
python train.py
```

### Using the Jupyter notebook

```bash
jupyter notebook
# Open train.ipynb
```

## Customizing Your Training

The training code is designed to be easily customizable:

1. **Change the base model**: Modify the `model_name` parameter in the script
2. **Change the dataset**: Replace the dataset loading code with your own dataset
3. **Adjust training parameters**: Modify batch size, learning rate, etc. in the `TrainingArguments`

## Saving and Using Your Model

After training, the model will be saved in the `outputs` directory. You can:

1. Load it directly using Hugging Face's Transformers library
2. Convert it to GGUF format for use with llama.cpp
3. Use it with Ollama for local inference

### Using with Ollama

1. Install Ollama from [ollama.com](https://ollama.com)
2. Convert your model to GGUF format (see script)
3. Create a Modelfile:
   ```
   FROM your_model.gguf
   SYSTEM <your system prompt>
   ```
4. Create an Ollama model:
   ```bash
   ollama create mymodel -f Modelfile
   ```
5. Run your model:
   ```bash
   ollama run mymodel
   ```

## Troubleshooting

- **Out of memory errors**: Reduce batch size or use a smaller model
- **CUDA errors**: Make sure you have the correct CUDA version installed
- **Import errors**: Make sure all dependencies are installed correctly

## Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Ollama Documentation](https://github.com/ollama/ollama)