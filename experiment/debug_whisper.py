#!/usr/bin/env python3
"""
Debug script to test Whisper model forward method
"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig
import inspect

def test_whisper_forward():
    """Test Whisper model forward method"""

    # Load model and processor
    model_name = "openai/whisper-large-v3-turbo"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Print model configuration
    print("Model configuration:")
    print(f"  - Model name: {model_name}")
    print(f"  - Model config: {model.config}")
    print(f"  - Expected input features shape: {getattr(model.config, 'num_mel_bins', 'Not specified')}")

    # Print the forward method signature
    print("\nOriginal Whisper forward signature:")
    print(inspect.signature(model.forward))
    
    # Create LoRA config (without task_type to avoid input_ids error)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        # task_type="SEQ_2_SEQ_LM",  # Remove this to avoid input_ids error
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    print("\nPEFT Whisper forward signature:")
    print(inspect.signature(model.forward))
    
    # Create dummy inputs using processor to get correct dimensions
    print(f"\nProcessor feature extractor config:")
    print(f"  - n_mels: {processor.feature_extractor.n_mels}")
    print(f"  - n_fft: {processor.feature_extractor.n_fft}")
    print(f"  - hop_length: {processor.feature_extractor.hop_length}")

    # Use processor to create properly formatted input
    dummy_audio = torch.randn(16000 * 2)  # 2 seconds of audio at 16kHz
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    print(f"\nActual input_features shape from processor: {input_features.shape}")

    batch_size = 2
    # Repeat for batch
    input_features = input_features.repeat(batch_size, 1, 1)
    labels = torch.randint(0, 1000, (batch_size, 20))
    
    print(f"\nInput shapes:")
    print(f"input_features: {input_features.shape}")
    print(f"labels: {labels.shape}")
    
    # Test forward call
    try:
        print("\nTesting forward call with input_features and labels...")
        outputs = model(input_features=input_features, labels=labels)
        print("✅ Forward call successful!")
        print(f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Forward call failed: {e}")

    # Test with input_ids instead of input_features (PEFT expects this)
    try:
        print("\nTesting forward call with input_ids and labels...")
        outputs = model(input_ids=input_features, labels=labels)
        print("✅ input_ids Forward call successful!")
        print(f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ input_ids Forward call failed: {e}")

    # Test with kwargs approach
    try:
        print("\nTesting forward call with kwargs (input_features)...")
        outputs = model(labels=labels, input_features=input_features)
        print("✅ kwargs Forward call successful!")
        print(f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ kwargs Forward call failed: {e}")

    # Test with different argument names
    try:
        print("\nTesting forward call with positional arguments...")
        outputs = model(input_features, labels)
        print("✅ Positional forward call successful!")
    except Exception as e:
        print(f"❌ Positional forward call failed: {e}")

if __name__ == "__main__":
    test_whisper_forward()
