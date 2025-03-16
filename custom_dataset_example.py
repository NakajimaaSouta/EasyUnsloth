"""
Example script for using a custom dataset with Unsloth

This script demonstrates how to:
1. Load a custom dataset from a JSON file
2. Format it for fine-tuning
3. Train a model using this custom dataset
"""

import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ===== CONFIGURATION =====
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
OUTPUT_DIR = "outputs_custom"
CUSTOM_DATASET_PATH = "custom_dataset_example.json"

# Prompt template for instruction fine-tuning
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

# ===== LOAD CUSTOM DATASET =====
print("Loading custom dataset...")
with open(CUSTOM_DATASET_PATH, 'r', encoding='utf-8') as f:
    custom_data = json.load(f)

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(custom_data)
print(f"Loaded {len(dataset)} examples from custom dataset")

# Display a sample
print("\nSample from the dataset:")
print(dataset[0])

# ===== LOAD MODEL AND TOKENIZER =====
print("\nLoading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ===== FORMAT DATASET =====
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses = examples["response"]
    
    texts = []
    for instruction, response in zip(instructions, responses):
        text = PROMPT_TEMPLATE.format(instruction=instruction, response=response) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}

# Apply the formatting function to the dataset
print("Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Display a formatted example
print("\nFormatted example:")
print(formatted_dataset[0]["text"][:500] + "...")

# ===== SETUP TRAINER =====
print("\nSetting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,  # Small number of steps for this example
        learning_rate=2e-4,
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
print("\nStarting training...")
if torch.cuda.is_available():
    print(f"Training on: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU (this will be slow)")

trainer_stats = trainer.train()

# ===== SAVE THE MODEL =====
print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")

print("\n===== TRAINING COMPLETE =====")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")

# ===== TEST THE MODEL =====
print("\nTesting the model with a sample instruction...")
# Format the prompt
test_instruction = "Write a haiku about artificial intelligence."
prompt = PROMPT_TEMPLATE.format(instruction=test_instruction, response="")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the response part
response_start = generated_text.find("### Response:")
if response_start != -1:
    response = generated_text[response_start + len("### Response:"):].strip()
else:
    response = generated_text[len(prompt):].strip()

print(f"\nInstruction: {test_instruction}")
print(f"\nGenerated response:\n{response}")

print("\nDone! You've successfully fine-tuned a model on your custom dataset.")