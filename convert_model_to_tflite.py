"""
Convert Keras model (.h5) to TensorFlow Lite (.tflite) format for Android.

This script converts your invisink_model.h5 to a mobile-optimized .tflite file
that can be used in Android apps.
"""

import tensorflow as tf
import os
import sys

def convert_model_to_tflite(model_path='invisink_model.h5', output_path='invisink_model.tflite'):
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to the input .h5 model file
        output_path: Path where the .tflite file will be saved
    """
    print("=" * 60)
    print("InvisInk Model Converter - H5 to TensorFlow Lite")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print(f"   Please ensure the model file exists in the current directory.")
        sys.exit(1)
    
    print(f"\n📦 Loading model from: {model_path}")
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Display model summary
        print("\n📊 Model Summary:")
        model.summary()
        
        # Get input shape
        input_shape = model.input_shape
        print(f"\n📐 Input shape: {input_shape}")
        print(f"📐 Output shape: {model.output_shape}")
        
        # Convert to TensorFlow Lite
        print("\n🔄 Converting to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Optimize for size and performance
        print("⚙️  Applying optimizations (DEFAULT)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Set representative dataset for full quantization (if needed)
        # Uncomment below if you want to quantize to int8 (smaller size)
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the converted model
        print(f"\n💾 Saving converted model to: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Display file sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        converted_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - converted_size / original_size) * 100
        
        print("\n" + "=" * 60)
        print("✅ Conversion Successful!")
        print("=" * 60)
        print(f"📁 Original model size: {original_size:.2f} MB")
        print(f"📁 TensorFlow Lite size: {converted_size:.2f} MB")
        print(f"📉 Size reduction: {compression_ratio:.1f}%")
        print(f"\n📱 Your model is ready for Android!")
        print(f"   Copy '{output_path}' to your Android app's assets folder.")
        print("=" * 60)
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure TensorFlow is installed: pip install tensorflow")
        print("2. Check that the model file is valid")
        print("3. Ensure you have write permissions in the current directory")
        sys.exit(1)


if __name__ == "__main__":
    # Default paths
    model_path = 'invisink_model.h5'
    output_path = 'invisink_model.tflite'
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Convert the model
    convert_model_to_tflite(model_path, output_path)

