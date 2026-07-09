# ComfyUI-Unlimited-OCR

ComfyUI custom nodes for [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) by [Baidu](https://github.com/baidu/Unlimited-OCR) - a vision-language model for high-accuracy optical character recognition (OCR) and text extraction.

**Unlimited-OCR** is a one-shot long-horizon parsing model capable of extracting text from images with high accuracy.

## References

| Resource | Link |
|----------|------|
| **HuggingFace Model** | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) |
| **GitHub Repository** | [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) |
| **Paper** | [arXiv:2606.23050](https://arxiv.org/abs/2606.23050) |

## Features

- **Automatic Model Download**: Downloads the Unlimited-OCR model from HuggingFace on first use
- **High-Accuracy OCR**: Extract text from images with high accuracy
- **Text Localization**: Locate and extract text regions within images
- **Configurable Inference**: Adjustable max_length, temperature, no_repeat_ngram_size, ngram_window, crop_mode, base_size, and image_size
- **Model Caching**: Models are cached to avoid redundant loading across workflows
- **Debug Visualization**: Inspect model loading state and test OCR results

## Nodes

### Loader & Config

| Node | Display Name | Description |
|------|-------|-------------|
| **UnlimitedOCRLoader** | Load Unlimited OCR Model | Loads the Unlimited-OCR model from HuggingFace with configurable device, dtype, and trust_remote_code settings |
| **UnlimitedOCRConfig** | Configure Inference | Creates inference configuration with parameters for max_length, temperature, no_repeat_ngram_size, ngram_window, crop_mode, base_size, and image_size |

### Inference

| Node | Display Name | Description |
|------|-------|-------------|
| **UnlimitedOCRInference** | OCR / Text Extraction | Performs OCR on an image with a custom prompt to control the type of OCR output |

### Debug

| Node | Display Name | Description |
|------|-------|-------------|
| **UnlimitedOCRDebug** | Debug Model | Inspects model loading state and runs test OCR with visualized results |

## Installation

1. Clone the repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
    git clone https://github.com/PsychoLogicAu/ComfyUI-Unlimited-OCR.git
   cd ComfyUI-Unlimited-OCR
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

4. Load the model in your workflow:
   - Add a **Load Unlimited OCR Model** node
   - Configure model path (default: `baidu/Unlimited-OCR`), dtype, and trust_remote_code
   - The model will be downloaded from HuggingFace on first use

## Usage

### Basic OCR Workflow

1. Add a **Load Unlimited OCR Model** node and configure:
   - `model_path`: `baidu/Unlimited-OCR` (or a local path)
   - `dtype`: `bfloat16` (recommended), `float16`, or `float32`
   - `trust_remote_code`: Enable trust remote code (default: true)

2. (Optional) Add a **Configure Inference** node to customize:
   - `max_length`: Maximum output length (default: 32768)
   - `temperature`: Sampling temperature (default: 0.0)
   - `no_repeat_ngram_size`: No-repeat n-gram size (default: 35)
   - `ngram_window`: N-gram window size (default: 128)
   - `crop_mode`: Enable cropping (default: true)
   - `base_size`: Base image size (default: 1024)
    - `image_size`: Image size for processing (default: 640)

3. Add an **OCR / Text Extraction** node:
   - Connect the model output and your image
   - Set a custom `prompt` to control the type of OCR output (e.g., "document parsing.", "extract text.")
   - The node will extract text and return annotated images with bounding boxes

4. The node outputs:
   - `extracted_text`: Cleaned extracted text
   - `annotated_image`: Image with bounding boxes drawn
   - `debug_text`: Raw model output with bounding box annotations

### Document Understanding Workflow

Use the **OCR / Text Extraction** node to extract and understand document content:

1. Connect model and image inputs
2. Set the `prompt` to "document parsing." for general document understanding
3. The node processes the image and extracts all text content
4. Adjust `max_length` if processing complex documents

## Output Format

### OCR Result

The `extracted_text` output from the **OCR / Text Extraction** node contains the cleaned extracted text.

The `annotated_image` output contains the image with bounding boxes drawn on detected text regions.

The `debug_text` output contains the raw model output with bounding box annotations.

## Model Information

- **Model**: Baidu Unlimited-OCR
- **HuggingFace**: [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
- **Upstream Repo**: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- **Paper**: [Unlimited OCR Works](https://arxiv.org/abs/2606.23050)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- Pillow (PIL)
- torchvision
- torchtyping
- sentencepiece

## License

This project is provided as-is for use with ComfyUI. The Unlimited-OCR model is licensed under the MIT License. See the [HuggingFace model card](https://huggingface.co/baidu/Unlimited-OCR) for licensing details.

## Troubleshooting

- **Model download fails**: Check your internet connection and ensure you have access to the HuggingFace model
- **CUDA out of memory**: Reduce `max_new_tokens`
- **Poor OCR quality**: Try a higher resolution image, adjust `temperature`, or increase `max_new_tokens`

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## Image Attribution

Input image in examples: Kaldari, CC0, via Wikimedia Commons