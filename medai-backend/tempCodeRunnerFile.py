import os
import tensorflow as tf
import skimage.segmentation
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import logging
import cv2
import base64
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedAI API", description="API for disease detection using AI models")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths with proper OS-specific path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

# Define full paths to model files
BLINDNESS_MODEL_PATH = os.path.join(MODEL_DIR, "blindness_model.h5")
BRAIN_TUMOR_MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor.h5")
PNEUMONIA_MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_detection_Vision_Model.h5")

# Print paths for debugging
logger.info(f"Looking for models in: {MODEL_DIR}")
logger.info(f"Blindness model path: {BLINDNESS_MODEL_PATH}")
logger.info(f"Brain tumor model path: {BRAIN_TUMOR_MODEL_PATH}")
logger.info(f"Pneumonia model path: {PNEUMONIA_MODEL_PATH}")

# Check if files exist
logger.info(f"Blindness model exists: {os.path.exists(BLINDNESS_MODEL_PATH)}")
logger.info(f"Brain tumor model exists: {os.path.exists(BRAIN_TUMOR_MODEL_PATH)}")
logger.info(f"Pneumonia model exists: {os.path.exists(PNEUMONIA_MODEL_PATH)}")

# XAI Utility Functions
def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Generate proper Grad-CAM visualization for pneumonia detection
    """
    try:
        # Find the last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                # Look for convolutional layers
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            logger.error("Could not find any convolutional layer in the model")
            return generate_gradcam_alternative(model, img_array)
            
        # Get original image
        if hasattr(img_array, 'numpy'):
            orig_img = img_array[0].numpy()
        else:
            orig_img = img_array[0]
            
        # Create a model that outputs both the last conv layer and the final output
        try:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[
                    model.get_layer(last_conv_layer_name).output,
                    model.output
                ]
            )
            
            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Cast inputs to float32
                inputs = tf.cast(img_array, tf.float32)
                
                # Forward pass to get conv output and model prediction
                conv_outputs, predictions = grad_model(inputs)
                
                # Get the predicted class (for pneumonia, this is the output probability)
                pred_index = 0 if predictions[0][0] < 0.5 else 0
                class_channel = predictions[:, pred_index]
                
            # Gradient of the predicted class with respect to the output feature map
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Vector of mean intensity of gradient over feature map
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by corresponding gradients
            conv_outputs = conv_outputs[0]
            for i in range(pooled_grads.shape[0]):
                conv_outputs[:, :, i] *= pooled_grads[i]
                
            # Average all channels
            heatmap = tf.reduce_mean(conv_outputs, axis=-1)
            
            # ReLU thresholding to only show positive contributions
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to match input image size
            heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
            
            # Convert heatmap to RGB using colormap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on original image
            img_rgb = np.uint8(orig_img * 255) if orig_img.max() <= 1.0 else np.uint8(orig_img)
            if len(img_rgb.shape) == 2:  # If grayscale
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
            elif img_rgb.shape[2] == 3:  # If RGB
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Create superimposed image with proper weighting
            superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
            
            return superimposed_img
        except Exception as e:
            logger.error(f"Error in Grad-CAM model creation: {e}")
            return generate_gradcam_alternative(model, img_array)
            
    except Exception as e:
        logger.error(f"Error in Grad-CAM generation: {e}")
        # Fallback to simpler method if the gradient approach fails
        return generate_gradcam_alternative(model, img_array)

def generate_gradcam_alternative(model, img_array):
    """
    Alternative Grad-CAM implementation for pneumonia model
    Based on intensity-based heatmap
    """
    # Get the prediction
    prediction = model.predict(img_array)
    pred_class = 1 if prediction[0][0] > 0.5 else 0
    
    # Convert tensor to numpy if needed
    if hasattr(img_array, 'numpy'):
        img = img_array[0].numpy()
    else:
        img = img_array[0]
    
    # Create a proper heatmap based on image intensity
    img_uint8 = np.uint8(img*255) if img.max() <= 1.0 else np.uint8(img)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Normalize the grayscale image to 0-255
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to smooth the heatmap
    blurred = cv2.GaussianBlur(normalized, (9, 9), 0)
    
    # Create heatmap - for pneumonia, brighter areas are more important
    if pred_class == 1:  # If pneumonia
        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    else:  # If normal
        heatmap = cv2.applyColorMap(255 - blurred, cv2.COLORMAP_JET)
    
    # Superimpose on original
    img_rgb = np.uint8(img*255) if img.max() <= 1.0 else np.uint8(img)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    elif img_rgb.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Use a better blending ratio
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img

def explain_with_lime(model, image, preprocess_fn):
    """
    Generate LIME explanation for pneumonia detection model
    """
    try:
        # Create explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Ensure image is the right format
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Make sure image is in the right range (0-1)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Define segmentation function for better superpixels
        # This creates more meaningful segments in medical images
        segmentation_fn = lambda x: skimage.segmentation.quickshift(
            x, kernel_size=3, max_dist=6, ratio=0.5, convert2lab=True
        )
        
        # Create a wrapper function that returns probabilities for both classes
        def predict_fn(x):
            preds = model.predict(preprocess_fn(x.astype(np.float32)))
            # Convert single probability to two-class probability array
            # [p] -> [[1-p, p]]
            return np.hstack([1-preds, preds])
        
        # Get explanation with more samples for stability
        explanation = explainer.explain_instance(
            image,
            predict_fn,
            top_labels=2,  # Explain both classes
            hide_color=0,
            num_samples=1000,  # More samples for stability
            segmentation_fn=segmentation_fn  # Custom segmentation
        )
        
        # Get labels from explanation
        labels = explanation.top_labels
        
        # If no labels, return placeholder
        if len(labels) == 0:
            logger.error("No labels available in LIME explanation")
            return np.ones((224, 224, 3)) * 0.5, None
        
        # For pneumonia, use label 1 if available, otherwise use first label
        label_to_explain = 1 if 1 in labels else labels[0]
        
        # Get mask with more features and positive+negative contributions
        temp, mask = explanation.get_image_and_mask(
            label_to_explain,
            positive_only=False,  # Show both positive and negative contributions
            num_features=10,      # Show more features
            hide_rest=False
        )
        
        # Create visualization with more prominent boundaries
        lime_img = mark_boundaries(temp/255.0, mask, color=(1, 0, 0), outline_color=(0, 1, 0), mode='outer')
        
        return lime_img, explanation
    except Exception as e:
        logger.error(f"LIME visualization error: {e}")
        # Return a placeholder image
        placeholder = np.ones((224, 224, 3)) * 0.5
        return placeholder, None

def explain_with_shap(model, image, preprocess_fn, background_samples=10):
    """
    Generate SHAP explanation for pneumonia detection model
    """
    try:
        # Create background dataset
        background = np.random.random((background_samples, 224, 224, 3))
        background = preprocess_fn(background)
        
        # Process the image
        processed_image = preprocess_fn(np.expand_dims(image, axis=0))
        
        # Try GradientExplainer instead of DeepExplainer
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(processed_image)
        except Exception as e:
            logger.error(f"GradientExplainer failed: {e}")
            # Fallback to DeepExplainer
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(processed_image)
        
        # Create visualization (using matplotlib)
        plt.figure(figsize=(10, 6))
        
        # Handle different return formats from different SHAP versions
        if isinstance(shap_values, list):
            # For newer SHAP versions that return a list
            shap.image_plot(shap_values, -processed_image, show=False)
        else:
            # For older SHAP versions
            shap.image_plot([shap_values], -processed_image, show=False)
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convert to base64 for web display
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str, shap_values
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return None, None

# Helper function to convert numpy image to base64
def image_to_base64(img):
    """Convert image to base64 string for web display"""
    if img is None:
        return None
        
    if isinstance(img, np.ndarray):
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Convert from BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Encode image
        success, encoded_img = cv2.imencode('.png', img)
        if success:
            return base64.b64encode(encoded_img).decode('utf-8')
    
    # For matplotlib figures already in memory
    elif isinstance(img, str):
        return img
    
    return None

def generate_pneumonia_explanation(predicted_class, confidence):
    """Generate textual explanation for pneumonia prediction"""
    if predicted_class == 1:
        if confidence > 0.9:
            return "The model has detected clear signs of pneumonia with high confidence. The highlighted areas show dense opacities in the lung fields, which are typical radiographic findings in pneumonia cases."
        elif confidence > 0.7:
            return "The model has detected moderate signs of pneumonia. The highlighted regions show areas of increased density that may represent pulmonary infiltrates consistent with pneumonia."
        else:
            return "The model has detected subtle signs that may indicate pneumonia, but with lower confidence. The highlighted areas show mild opacities that could represent early pneumonia or other conditions."
    else:
        if confidence > 0.9:
            return "The model has determined with high confidence that this X-ray shows normal lung fields without signs of pneumonia. The lung fields appear clear without significant opacities."
        elif confidence > 0.7:
            return "The model has classified this as a normal chest X-ray. The lung fields appear mostly clear, though there may be some normal anatomical structures highlighted."
        else:
            return "The model has classified this as likely normal, but with lower confidence. Some areas are highlighted that may represent normal anatomical structures or very subtle abnormalities."

# Define model architectures and load weights
logger.info("Loading models...")

# 1. Blindness Detection Model (DenseNet121)
def create_blindness_model():
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Initialize models
blindness_model = None
brain_tumor_model = None
pneumonia_model = None

# Create models and load weights if files exist
if os.path.exists(BLINDNESS_MODEL_PATH):
    try:
        blindness_model = create_blindness_model()
        blindness_model.load_weights(BLINDNESS_MODEL_PATH)
        logger.info("Blindness model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading blindness model: {e}")

# Load brain tumor model as a complete model (architecture + weights) from GitHub repo
if os.path.exists(BRAIN_TUMOR_MODEL_PATH):
    try:
        brain_tumor_model = tf.keras.models.load_model(BRAIN_TUMOR_MODEL_PATH)
        # Test prediction on a sample image to verify model is working
        test_img = np.random.random((1, 150, 150, 3))
        test_pred = brain_tumor_model.predict(test_img)
        logger.info(f"Brain tumor model test prediction shape: {test_pred.shape}")
        logger.info(f"Brain tumor model output classes: {np.argmax(test_pred[0])}")
        logger.info("Brain tumor model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading brain tumor model: {e}")

# Load pneumonia model as a complete model (architecture + weights)
if os.path.exists(PNEUMONIA_MODEL_PATH):
    try:
        pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
        logger.info("Pneumonia model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading pneumonia model: {e}")
        # Fallback to a simple model if loading fails
        try:
            logger.info("Attempting to load pneumonia model with a simpler architecture...")
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
            
            pneumonia_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            pneumonia_model.load_weights(PNEUMONIA_MODEL_PATH)
            logger.info("Pneumonia model loaded with simpler architecture")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")

# Check if any models were loaded
if not any([blindness_model, brain_tumor_model, pneumonia_model]):
    logger.error("No models were loaded successfully. Please check model paths and files.")

# Preprocessing function for general use
def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Get original image for display
def get_original_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    return np.array(image)

# Preprocessing function specifically for brain tumor images using OpenCV
def preprocess_brain_image(image_bytes, target_size=(150, 150)):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Resize
    img = cv2.resize(img, target_size)
    # Reshape to add batch dimension
    img = np.reshape(img, (1, 150, 150, 3))
    return img

@app.get("/")
def read_root():
    models_loaded = {
        "blindness_model": blindness_model is not None,
        "brain_tumor_model": brain_tumor_model is not None,
        "pneumonia_model": pneumonia_model is not None
    }
    return {
        "message": "Welcome to MedAI API", 
        "status": "active",
        "models_loaded": models_loaded
    }

@app.post("/predict/blindness")
async def predict_blindness(file: UploadFile = File(...)):
    if blindness_model is None:
        raise HTTPException(status_code=503, detail="Blindness detection model not loaded")
    
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        prediction = blindness_model.predict(processed_image)
        
        # Get the predicted class (0-4)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        # Map class to severity
        severity_map = {
            0: "No DR",
            1: "Mild DR",
            2: "Moderate DR", 
            3: "Severe DR",
            4: "Proliferative DR"
        }
        
        return {
            "prediction": predicted_class,
            "severity": severity_map[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in blindness prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/brain-tumor")
async def predict_brain_tumor(file: UploadFile = File(...)):
    if brain_tumor_model is None:
        raise HTTPException(status_code=503, detail="Brain tumor model not loaded")
    
    try:
        contents = await file.read()
        processed_image = preprocess_brain_image(contents)  # Use OpenCV preprocessing
        prediction = brain_tumor_model.predict(processed_image)
        
        # Log the raw prediction for debugging
        logger.info(f"Raw brain tumor prediction: {prediction[0]}")
        
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        # Log the predicted class and confidence
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        
        # Map class to tumor type (using GitHub repo's mapping)
        tumor_map = {
            0: "Glioma Tumor",
            1: "Meningioma Tumor",
            2: "No Tumor Found",
            3: "Pituitary Tumor"
        }
        
        return {
            "prediction": predicted_class,
            "tumor_type": tumor_map[predicted_class],
            "confidence": confidence,
            "raw_prediction": prediction[0].tolist()  # Include raw values
        }
    except Exception as e:
        logger.error(f"Error in brain tumor prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    if pneumonia_model is None:
        raise HTTPException(status_code=503, detail="Pneumonia model not loaded")
    
    try:
        contents = await file.read()
        
        # Get original image for display and XAI
        original_image = get_original_image(contents)
        original_image_normalized = original_image / 255.0  # Normalize for visualization
        
        # Preprocess for model
        processed_image = preprocess_image(contents)
        
        # Get prediction
        prediction = pneumonia_model.predict(processed_image)
        
        # Extract prediction details
        raw_value = float(prediction[0][0]) if prediction.shape[1:] == (1,) else float(prediction[0])
        logger.info(f"Raw pneumonia prediction value: {raw_value}")
        
        predicted_class = 1 if raw_value > 0.5 else 0
        confidence = raw_value if predicted_class == 1 else 1 - raw_value
        result = "Pneumonia" if predicted_class == 1 else "Normal"
        
        # Generate explanations
        try:
            # Try the proper Grad-CAM first, fall back to alternative if needed
            gradcam_img = generate_gradcam(pneumonia_model, processed_image)
            if gradcam_img is None:
                gradcam_img = generate_gradcam_alternative(pneumonia_model, processed_image)
            gradcam_img_b64 = image_to_base64(gradcam_img)
            
            # Try improved LIME explanation
            lime_img, _ = explain_with_lime(pneumonia_model, original_image_normalized, 
                                          lambda x: np.expand_dims(x, axis=0) if len(x.shape) == 3 else x)
            lime_img_b64 = image_to_base64((lime_img * 255).astype(np.uint8))
            
            # Try SHAP explanation - often heavy on memory, so wrap in try/except
            try:
                shap_img_b64, _ = explain_with_shap(pneumonia_model, original_image_normalized, 
                                                  lambda x: np.expand_dims(x, axis=0) if len(x.shape) == 3 else x)
            except Exception as shap_error:
                logger.error(f"SHAP explanation failed: {shap_error}")
                shap_img_b64 = None
            
            # Generate textual explanation
            explanation_text = generate_pneumonia_explanation(predicted_class, confidence)
            
            # Original image as base64
            original_img_b64 = image_to_base64(original_image)
            
            # Ensure we have at least some explanations
            explanations_available = True
            
        except Exception as xai_error:
            logger.error(f"Error generating explanations: {xai_error}")
            explanations_available = False
            original_img_b64 = image_to_base64(original_image)
            gradcam_img_b64 = None
            lime_img_b64 = None
            shap_img_b64 = None
            explanation_text = "Explanations could not be generated due to an error."
        
        # Build response with available explanations
        response = {
            "prediction": predicted_class,
            "result": result,
            "confidence": confidence,
            "raw_value": raw_value
        }
        
        # Always include explanation object, even if some visualizations failed
        response["explanation"] = {
            "text": explanation_text,
            "original_image": original_img_b64
        }
        
        # Add available visualizations
        if gradcam_img_b64 is not None:
            response["explanation"]["gradcam_image"] = gradcam_img_b64
        if lime_img_b64 is not None:
            response["explanation"]["lime_image"] = lime_img_b64
        if shap_img_b64 is not None:
            response["explanation"]["shap_image"] = shap_img_b64
        
        return response
        
    except Exception as e:
        logger.error(f"Error in pneumonia prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
