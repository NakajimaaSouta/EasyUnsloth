"""
Unsloth Fine-Tuning Script

This script demonstrates how to fine-tune a language model using Unsloth.
It's designed to be easy to understand and customize.
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ===== CONFIGURATION =====
# Feel free to modify these parameters based on your needs

# Model configuration
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Pre-quantized model for faster loading
MAX_SEQ_LENGTH = 2048  # Context length
LOAD_IN_4BIT = True    # Use 4-bit quantization to reduce memory usage

# LoRA configuration (Low-Rank Adaptation for efficient fine-tuning)
LORA_RANK = 16         # Rank of LoRA matrices (higher = more capacity, more memory)
LORA_ALPHA = 16        # LoRA alpha parameter
LORA_DROPOUT = 0       # LoRA dropout (0 is optimized)

# Training configuration
BATCH_SIZE = 2         # Batch size per device
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over multiple steps
LEARNING_RATE = 2e-4   # Learning rate
MAX_STEPS = 100        # Number of training steps (set to None to train for full epochs)
NUM_EPOCHS = None      # Number of epochs (alternative to max_steps)
OUTPUT_DIR = "outputs" # Directory to save the model

# ===== PROMPT TEMPLATE =====
# This template formats your data for instruction fine-tuning
# Modify this based on your specific use case

PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

# ===== LOAD MODEL AND TOKENIZER =====
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect dtype
    load_in_4bit=LOAD_IN_4BIT,
    # token="hf_...",  # Uncomment and add your token if using gated models
)

# Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
print("Applying LoRA for efficient fine-tuning...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",  # "none" is optimized
    use_gradient_checkpointing="unsloth",  # "unsloth" uses 30% less VRAM
    random_state=42,
    use_rslora=False,  # Rank-stabilized LoRA (optional)
    loftq_config=None, # LoftQ (optional)
)

# ===== LOAD AND PREPARE DATASET =====
# This example uses the Alpaca dataset, but you can replace it with your own
print("Loading dataset...")

# Option 1: Load a dataset from Hugging Face
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# Alternatively, you can load your own dataset:
# dataset = load_dataset("json", data_files="your_data.json", split="train")

# Function to format the prompts
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses = examples["output"]
    
    texts = []
    for instruction, response in zip(instructions, responses):
        # Format the text according to the template and add EOS token
        text = PROMPT_TEMPLATE.format(instruction=instruction, response=response) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}

# Apply the formatting function to the dataset
print("Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# ===== SETUP TRAINER =====
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,  # Number of processes for dataset preparation
    packing=False,  # Set to True for short sequences to speed up training
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
    ),
)

# ===== TRAIN THE MODEL =====
print("Starting training...")
# Print GPU information
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name}")
    print(f"Total GPU memory: {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 2)} GB")
    
# Track memory usage
start_gpu_memory = 0
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Starting GPU memory usage: {start_gpu_memory} GB")

# Train the model
trainer_stats = trainer.train()

# Print training statistics
print("\n===== TRAINING COMPLETE =====")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")

# Print memory usage
if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    print(f"Peak memory usage: {used_memory} GB")
    print(f"Memory used for training: {used_memory_for_training} GB")

# ===== SAVE THE MODEL =====
print("\nSaving model...")
# Save the LoRA adapter only (small file)
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")

# Optional: Save the merged model (much larger file)
print("\nSaving merged model (this may take a while)...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_model", tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {OUTPUT_DIR}/merged_model")

print("\n===== NEXT STEPS =====")
print("1. To convert to GGUF format for llama.cpp or Ollama:")
print("   python -m unsloth.convert_to_gguf outputs/merged_model --outfile my_model.gguf")
print("2. To use with Ollama:")
print("   ollama create mymodel -f Modelfile")
print("3. To test your model:")
print("   ollama run mymodel")

# Clean up CUDA memory
torch.cuda.empty_cache()
