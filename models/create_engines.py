#!/usr/bin/env python3
"""
Create TensorRT engines for P3: Model Variation experiment.
Models: MiniLM-L6, MiniLM-L12, DistilBERT, BERT-base
"""

import os
import subprocess
import sys

# Check dependencies
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import onnx
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch transformers onnx onnxruntime")
    sys.exit(1)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "minilm_l6": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "description": "MiniLM-L6-v2 (22M params, 6 layers)"
    },
    "minilm_l12": {
        "hf_name": "sentence-transformers/all-MiniLM-L12-v2",
        "max_length": 256,
        "description": "MiniLM-L12-v2 (33M params, 12 layers)"
    },
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "max_length": 256,
        "description": "DistilBERT (66M params, 6 layers)"
    },
    "bert_base": {
        "hf_name": "bert-base-uncased",
        "max_length": 256,
        "description": "BERT-base (110M params, 12 layers)"
    }
}

def export_to_onnx(model_key):
    """Export HuggingFace model to ONNX format."""
    config = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Exporting {config['description']} to ONNX...")
    print(f"{'='*60}")

    onnx_path = os.path.join(MODELS_DIR, f"{model_key}.onnx")

    if os.path.exists(onnx_path):
        print(f"ONNX file already exists: {onnx_path}")
        return onnx_path

    # Load model and tokenizer
    print(f"Loading model: {config['hf_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
    model = AutoModel.from_pretrained(config['hf_name'])
    model.eval()

    # Create dummy input
    dummy_text = "This is a test sentence for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt",
                       max_length=config['max_length'],
                       padding="max_length", truncation=True)

    # Export to ONNX
    print(f"Exporting to: {onnx_path}")
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print(f"ONNX export complete: {onnx_path}")
    return onnx_path

def convert_to_tensorrt(model_key):
    """Convert ONNX model to TensorRT engine."""
    onnx_path = os.path.join(MODELS_DIR, f"{model_key}.onnx")
    engine_path = os.path.join(MODELS_DIR, f"{model_key}.engine")

    print(f"\n{'='*60}")
    print(f"Converting {model_key} to TensorRT...")
    print(f"{'='*60}")

    if os.path.exists(engine_path):
        print(f"Engine already exists: {engine_path}")
        return engine_path

    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        return None

    # Use trtexec for conversion
    max_len = MODELS[model_key]['max_length']
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=input_ids:1x1,attention_mask:1x1",
        f"--optShapes=input_ids:1x64,attention_mask:1x64",
        f"--maxShapes=input_ids:1x{max_len},attention_mask:1x{max_len}",
        "--workspace=4096"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"TensorRT conversion failed:")
        print(result.stderr)
        return None

    print(f"TensorRT engine created: {engine_path}")
    return engine_path

def main():
    print("P3: Creating TensorRT engines for Model Variation experiment")
    print(f"Models directory: {MODELS_DIR}")

    # Process each model
    for model_key in MODELS:
        try:
            # Export to ONNX
            onnx_path = export_to_onnx(model_key)

            # Convert to TensorRT
            if onnx_path:
                convert_to_tensorrt(model_key)
        except Exception as e:
            print(f"Error processing {model_key}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for model_key in MODELS:
        engine_path = os.path.join(MODELS_DIR, f"{model_key}.engine")
        status = "OK" if os.path.exists(engine_path) else "MISSING"
        print(f"  {model_key}: {status}")

if __name__ == "__main__":
    main()
