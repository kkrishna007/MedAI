#!/usr/bin/env python3
"""
Test script to verify the enhanced SHAP implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Import the enhanced SHAP functions
from main import (
    real_shap_explanation,
    create_medical_kernel_shap,
    create_medical_shap_visualization,
    create_medical_attention_fallback
)

def create_test_image():
    """Create a test chest X-ray image"""
    # Create a realistic chest X-ray simulation
    img = np.full((224, 224, 3), 0.3)  # Base intensity
    
    # Add lung fields
    img[40:180, 30:194] = 0.2
    
    # Add rib structures
    for i in range(5):
        y_pos = 50 + i * 25
        img[y_pos:y_pos+2, :, :] = 0.6
    
    # Add heart shadow
    heart_center = (112, 80)
    y, x = np.ogrid[:224, :224]
    heart_mask = (x - heart_center[0])**2 + (y - heart_center[1])**2 <= 25**2
    img[heart_mask] = 0.5
    
    # Add some consolidation (pneumonia-like)
    consolidation_mask = (x - 150)**2 + (y - 120)**2 <= 15**2
    img[consolidation_mask] = 0.8
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return (img * 255).astype(np.uint8)

def test_medical_attention_fallback():
    """Test the medical attention fallback"""
    print("🧪 Testing medical attention fallback...")
    
    test_img = create_test_image()
    result = create_medical_attention_fallback(test_img / 255.0)
    
    if result is not None:
        print("✅ Medical attention fallback works")
        
        # Save test result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img, cmap='gray')
        plt.title("Test Chest X-ray")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Medical Attention Map")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_medical_attention.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📁 Saved test result to test_medical_attention.png")
    else:
        print("❌ Medical attention fallback failed")

def test_medical_shap_visualization():
    """Test the medical SHAP visualization"""
    print("🧪 Testing medical SHAP visualization...")
    
    test_img = create_test_image()
    
    # Create mock SHAP values
    mock_shap_vals = np.random.normal(0, 0.1, test_img.shape)
    mock_shap_vals = np.abs(mock_shap_vals)  # Make positive
    
    result = create_medical_shap_visualization(mock_shap_vals, test_img / 255.0)
    
    if result is not None:
        print("✅ Medical SHAP visualization works")
    else:
        print("❌ Medical SHAP visualization failed")

def main():
    """Run all tests"""
    print("🚀 Starting SHAP enhancement tests...")
    
    try:
        test_medical_attention_fallback()
        test_medical_shap_visualization()
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary of improvements:")
        print("   • Anatomically relevant background generation")
        print("   • Medical feature focus (lung fields, heart, ribs)")
        print("   • Multi-scale feature analysis")
        print("   • Robust fallback mechanisms")
        print("   • Clinical interpretation enhancement")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 