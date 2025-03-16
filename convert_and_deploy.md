# Converting and Deploying Your Fine-Tuned Model

This guide explains how to convert your fine-tuned model to GGUF format and deploy it using Ollama for local inference.

## What is GGUF?

GGUF (GPT-Generated Unified Format) is a file format used by llama.cpp and other inference engines to run language models efficiently on consumer hardware. It's designed to be:

- Fast for inference
- Memory-efficient
- Portable across different platforms
- Supports various quantization levels

## Converting to GGUF Format

After fine-tuning your model, you'll need to convert it to GGUF format for use with llama.cpp or Ollama.

### Method 1: Using Unsloth's Built-in Converter

Unsloth provides a built-in converter that makes this process simple:

```bash
# Make sure you've saved the merged model first
python -m unsloth.convert_to_gguf outputs/merged_model --outfile my_model.gguf
```

### Method 2: Using llama.cpp's Converter

Alternatively, you can use llama.cpp's converter script:

```bash
# Clone llama.cpp repository
git clone --recursive https://github.com/ggerganov/llama.cpp

# Build llama.cpp
cd llama.cpp
make clean
make all -j

# Install required packages
pip install gguf protobuf

# Convert the model
python llama.cpp/convert_hf_to_gguf.py outputs/merged_model --outfile my_model.gguf --outtype q8_0
```

The `--outtype` parameter specifies the quantization level:
- `f16`: 16-bit floating point (largest file, highest quality)
- `q8_0`: 8-bit quantization (good balance of quality and size)
- `q4_0`: 4-bit quantization (smallest file, lower quality)

## Deploying with Ollama

[Ollama](https://ollama.com) is a tool that makes it easy to run language models locally.

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com).

### 2. Create a Modelfile

Create a file named `Modelfile` with the following content:

```
FROM my_model.gguf

SYSTEM You are a helpful assistant that provides accurate and informative responses.

# Optional: Add a template for chat-style interactions
TEMPLATE """
{{ if .System }}{{ .System }}{{ end }}

{{ range .Messages }}
{{- if eq .Role "user" }}
USER: {{ .Content }}
{{- else if eq .Role "assistant" }}
ASSISTANT: {{ .Content }}
{{- end }}
{{ end }}

USER: {{ .Prompt }}
ASSISTANT: 
"""

# Optional: Set parameters for generation
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
```

Customize the `SYSTEM` prompt based on your model's purpose.

### 3. Create the Ollama Model

Run the following command to create an Ollama model:

```bash
ollama create mymodel -f Modelfile
```

Replace `mymodel` with your preferred name.

### 4. Run the Model

Start using your model:

```bash
ollama run mymodel
```

You can now interact with your fine-tuned model through the command line.

## Advanced: Integrating with Applications

### Python API

You can use Ollama's API to integrate your model with Python applications:

```python
import requests

def generate_response(prompt):
    response = requests.post('http://localhost:11434/api/generate', 
                           json={
                               'model': 'mymodel',
                               'prompt': prompt,
                               'stream': False
                           })
    return response.json()['response']

# Example usage
result = generate_response("Explain quantum computing in simple terms.")
print(result)
```

### Web UI

You can use projects like [ollama-webui](https://github.com/ollama-webui/ollama-webui) to provide a ChatGPT-like interface for your model:

```bash
# Clone the repository
git clone https://github.com/ollama-webui/ollama-webui.git

# Navigate to the directory
cd ollama-webui

# Start the application
docker-compose up -d
```

Then open your browser and navigate to `http://localhost:3000`.

## Performance Tips

1. **Quantization Level**: Choose the appropriate quantization level based on your hardware and quality requirements.
2. **Context Length**: Be mindful of the context length used during training and inference.
3. **GPU Acceleration**: Ollama automatically uses GPU if available, which significantly improves performance.
4. **Memory Usage**: Lower quantization levels (q4_0, q5_0) use less memory but may reduce quality.

## Troubleshooting

- **Model Not Loading**: Ensure the GGUF file path in the Modelfile is correct.
- **Slow Performance**: Try a lower quantization level or reduce the context length.
- **Out of Memory**: Use a more aggressive quantization level or run on a machine with more RAM.
- **Poor Quality Outputs**: Try a higher quantization level or check if your fine-tuning process was successful.