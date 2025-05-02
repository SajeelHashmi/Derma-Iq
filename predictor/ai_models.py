import os
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from django.conf import settings

class AIWorker:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIWorker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Skip initialization if already initialized
        if AIWorker._initialized:
            return
            
        base_path = settings.BASE_DIR  
        models_dir = os.path.join(base_path, 'predictor', 'Models')
        self.general_disease_model_path = os.path.join(models_dir, 'UNET_PLUS_dataset1.keras')
        self.specific_acne_model_path = os.path.join(models_dir, 'ACNE-04_UNET_Plus.keras')
        self.face_isolation_yolo_model_path = os.path.join(models_dir, 'YOLO_FACE_BEST.pt')
        
        # Thresholds for binary mask generation
        # These would be determined from previous validation
        self.thresholds_general_diseases = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Default thresholds
        self.threshold_acne = 0.5  # Default threshold
        
        print("Loading models...")
        self.general_disease_model = self.load_model(self.general_disease_model_path)
        self.specific_acne_model = self.load_model(self.specific_acne_model_path)
        self.face_isolation_yolo_model = YOLO(self.face_isolation_yolo_model_path)
        print("Models loaded successfully")
        
        # Mark as initialized
        AIWorker._initialized = True
    
    def load_model(self, model_path):
        """Load a Keras model from the specified path"""
        return tf.keras.models.load_model(model_path)
    
    def preprocess_image(self, image: np.array) -> np.array:
        """
        Preprocess the image for model input.
        Returns a numpy array of shape (1, 256, 256, 1)
        """
        # Resize the image to 256x256
        image = cv2.resize(image, (256, 256))
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Equalize the histogram
        image = cv2.equalizeHist(image)
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
   
    def predict(self, image: np.array) -> dict:
        """
        Process the input image through the pipeline of models.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing:
            - original_image: The input image
            - face_image: The isolated face image
            - general_disease_raw_masks: Raw prediction masks from general model
            - general_disease_binary_masks: Thresholded binary masks for each disease class
            - specific_acne_raw_mask: Raw prediction mask from acne model
            - specific_acne_binary_mask: Thresholded binary mask for acne
        """
        # Ensure we have a valid image
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            raise ValueError("Invalid input image")
        
        # Store the original image for reference
        original_image = image.copy()
        face = None
        
        # Isolate Face using YOLO model
        face_yolo_res = self.face_isolation_yolo_model(image)
        
        # Extract face from bounding box
        for result in face_yolo_res:
            if len(result.boxes) > 0:
                # Use the first detected face
                bbox = result.boxes.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox)
                face = image[y1:y2, x1:x2]
                break
        
        # If no face detected, use the entire image
        if face is None or (isinstance(face, np.ndarray) and face.size == 0):
            print("No face detected, using the entire image")
            face = original_image
        
        # Preprocess the isolated face image for model input
        preprocessed_image = self.preprocess_image(face)
        
        # Predict using the general disease model
        # This model returns 10 segmentation masks for 10 diseases
        predictions_general = self.general_disease_model.predict(preprocessed_image)
        print(f"General disease predictions shape: {predictions_general.shape}")
        
        # Predict using the specific acne model
        # This model returns 1 segmentation mask for acne disease
        predictions_acne = self.specific_acne_model.predict(preprocessed_image)
        print(f"Specific acne prediction shape: {predictions_acne.shape}")
        
        # Create binary masks using the thresholds
        general_disease_binary_masks = []
        
        # Process general disease predictions
        for i in range(predictions_general.shape[-1]):
            # Apply threshold to create binary mask
            threshold = self.thresholds_general_diseases[i]
            binary_mask = (predictions_general[0, ..., i] > threshold).astype(np.uint8) * 255
            general_disease_binary_masks.append(binary_mask)
            
        # Process acne prediction
        acne_binary_mask = (predictions_acne[0, ..., 0] > self.threshold_acne).astype(np.uint8) * 255
        
        results = {
            "original_image": original_image,
            "face_image": face,
            "general_disease_raw_masks": predictions_general[0],  # Remove batch dimension
            "general_disease_binary_masks": general_disease_binary_masks,
            "specific_acne_raw_mask": predictions_acne[0, ..., 0],  # Remove batch and channel dimensions
            "specific_acne_binary_mask": acne_binary_mask
        }
        
        return results