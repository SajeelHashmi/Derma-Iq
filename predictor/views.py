import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.http import JsonResponse
from .ai_models import AIWorker


# intializing to allow preloading of models
ai_pipe = AIWorker()
def predict(request):
    if request.method == 'POST':
        # Get uploaded images
        face_frontal = request.FILES.get("face_frontal")
        face_left = request.FILES.get("face_left")
        face_right = request.FILES.get("face_right")
        faces = [face_frontal, face_left, face_right]
        
        # Validate that all images are uploaded
        if not all(faces):
            return render(request, 'predict.html', context={"error": "Please upload all three images."})
        
        try:
            # Get the singleton instance of AIWorker
            ai_worker = AIWorker()
            results = {}
            
            # Process frontal face image
            face_frontal_img = Image.open(face_frontal).convert('RGB')
            face_frontal_array = np.array(face_frontal_img)
            
            # Process left face image
            face_left_img = Image.open(face_left).convert('RGB')
            face_left_array = np.array(face_left_img)
            
            # Process right face image
            face_right_img = Image.open(face_right).convert('RGB')
            face_right_array = np.array(face_right_img)
            
            # Get predictions for each face
            results['face_frontal'] = ai_worker.predict(face_frontal_array)
            results['face_left'] = ai_worker.predict(face_left_array)
            results['face_right'] = ai_worker.predict(face_right_array)
            
            # Generate visualizations for UI display
            visualizations = generate_visualizations(results)
            
            # Prepare context for the template
            context = {
                "success": True,
                "visualizations": visualizations,
                "results": process_results_for_template(results)
            }
            
            return render(request, 'results.html', context=context)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing images: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return render(request, 'predict.html', context={"error": error_msg})
    
    # GET request: show the upload form
    return render(request, 'predict.html',context={'angles':["frontal", "left", "right"]})

def generate_visualizations(results):
    """Generate visualizations for the results to display in the UI"""
    visualizations = {}
    
    # Process each face direction
    for face_direction, face_results in results.items():
        face_viz = {}
        
        # Original face image
        face_image = face_results["face_image"]
        face_viz["face"] = image_to_base64(face_image)
        
        # General disease masks
        general_binary_masks = face_results["general_disease_binary_masks"]
        general_viz = []
        
        for i, mask in enumerate(general_binary_masks):
            mask_image = image_to_base64(mask)
            general_viz.append({
                "class_name": f"Disease Class {i}",
                "image": mask_image
            })
        
        face_viz["general_diseases"] = general_viz
        
        # Acne mask
        acne_mask = face_results["specific_acne_binary_mask"]
        face_viz["acne"] = image_to_base64(acne_mask)
        
        visualizations[face_direction] = face_viz
    
    return visualizations

def image_to_base64(image):
    """Convert numpy array image to base64 for HTML display using PIL"""

    buf = io.BytesIO()
    
    # Convert numpy array to PIL Image
    if len(image.shape) == 2:  # If grayscale
        pil_img = Image.fromarray(image.astype('uint8'), 'L')
    else:  # If RGB
        # Ensure image is in the correct format for PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        if image.shape[2] == 3:  # RGB
            pil_img = Image.fromarray(image, 'RGB')
        elif image.shape[2] == 4:  # RGBA
            pil_img = Image.fromarray(image, 'RGBA')
        else:
            # Convert to RGB
            from PIL import ImageOps
            pil_img = Image.fromarray(image[:,:,0].astype('uint8'), 'L')
            pil_img = ImageOps.colorize(pil_img, (0, 0, 0), (255, 255, 255))
    
    # Save image to buffer
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"




def process_results_for_template(results):
    """Process the raw prediction results for template rendering"""
    processed_results = {}
    
    # Calculate metrics for each face direction
    for face_direction, face_results in results.items():
        # Calculate percentage of affected area for each disease class
        general_disease_percentages = []
        
        for i, mask in enumerate(face_results["general_disease_binary_masks"]):
            percentage = (np.sum(mask > 0) / mask.size) * 100
            general_disease_percentages.append({
                "class_name": f"Disease Class {i}",
                "affected_percentage": round(percentage, 2)
            })
        
        # Calculate acne affected percentage
        acne_mask = face_results["specific_acne_binary_mask"]
        acne_percentage = (np.sum(acne_mask > 0) / acne_mask.size) * 100
        
        processed_results[face_direction] = {
            "general_diseases": general_disease_percentages,
            "acne": {"affected_percentage": round(acne_percentage, 2)}
        }
    
    return processed_results