"""
HuggingFace Deepfake Detector
Real pre-trained model for deepfake detection

Installation:
pip install transformers torch torchvision pillow

Usage:
from huggingface_detector import HuggingFaceDeepfakeDetector
detector = HuggingFaceDeepfakeDetector()
result = detector.predict('image.jpg')
"""

from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceDeepfakeDetector:
    """
    Real deepfake detection using pre-trained models from HuggingFace
    
    Supports multiple pre-trained models:
    1. dima806/deepfake_vs_real_image_detection - Good general purpose
    2. abhinavtripathi/deepfake-detection - Alternative
    3. rizvandwiki/deepfakes-image-detection - Another option
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the detector
        
        Args:
            model_name: HuggingFace model name. If None, tries multiple models.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # List of available models to try
        self.available_models = [
            "dima806/deepfake_vs_real_image_detection",
            "abhinavtripathi/deepfake-detection",
            "rizvandwiki/deepfakes-image-detection"
        ]
        
        self.model = None
        self.processor = None
        self.loaded = False
        
        # Try to load model
        if model_name:
            self._load_model(model_name)
        else:
            # Try each model until one works
            for model_name in self.available_models:
                if self._load_model(model_name):
                    break
    
    def _load_model(self, model_name):
        """Load a specific model"""
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            self.model_name = model_name
            logger.info(f"✓ Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            return False
    
    def predict(self, image_path):
        """
        Predict if an image is a deepfake
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with prediction results
        """
        if not self.loaded:
            logger.error("No model loaded!")
            return {
                'is_deepfake': False,
                'fake_probability': 50.0,
                'real_probability': 50.0,
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # Get probabilities
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()
            
            # Determine prediction
            is_deepfake = fake_prob > 0.5
            confidence = max(real_prob, fake_prob) * 100
            
            result = {
                'is_deepfake': bool(is_deepfake),
                'fake_probability': float(fake_prob * 100),
                'real_probability': float(real_prob * 100),
                'confidence': float(confidence),
                'model_used': self.model_name
            }
            
            logger.info(f"Prediction: {'FAKE' if is_deepfake else 'REAL'} ({confidence:.1f}% confident)")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'is_deepfake': False,
                'fake_probability': 50.0,
                'real_probability': 50.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_from_array(self, image_array):
        """
        Predict from numpy array (for integration with OpenCV)
        
        Args:
            image_array: numpy array (H, W, C) in BGR format
            
        Returns:
            dict with prediction results
        """
        if not self.loaded:
            return {
                'is_deepfake': False,
                'fake_probability': 50.0,
                'real_probability': 50.0,
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Convert BGR to RGB
            import cv2
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            
            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()
            is_deepfake = fake_prob > 0.5
            confidence = max(real_prob, fake_prob) * 100
            
            return {
                'is_deepfake': bool(is_deepfake),
                'fake_probability': float(fake_prob * 100),
                'real_probability': float(real_prob * 100),
                'confidence': float(confidence),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'is_deepfake': False,
                'fake_probability': 50.0,
                'real_probability': 50.0,
                'confidence': 0.0,
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    # Initialize detector
    print("Initializing detector...")
    detector = HuggingFaceDeepfakeDetector()
    
    if detector.loaded:
        print(f"✓ Detector ready! Using model: {detector.model_name}")
        print(f"Device: {detector.device}")
        
        # Test prediction
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            print(f"\nTesting with {test_image}...")
            result = detector.predict(test_image)
            
            print("\nResults:")
            print(f"  Is Deepfake: {result['is_deepfake']}")
            print(f"  Fake Probability: {result['fake_probability']:.2f}%")
            print(f"  Real Probability: {result['real_probability']:.2f}%")
            print(f"  Confidence: {result['confidence']:.2f}%")
        else:
            print(f"Test image not found: {test_image}")
    else:
        print("✗ Failed to load detector")