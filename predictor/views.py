import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render , redirect
from django.contrib import messages
from django.http import JsonResponse
from .ai_models import AIWorker
from .models import Result
import cv2
import numpy as np

# intializing to allow preloading of models
ai_pipe = AIWorker()

def predict(request):
    if request.user.is_authenticated:
        user = request.user
    else:
        messages.error(request, "Sign Up and begin Scanning.")
        return redirect('signup')
    if request.method == 'POST':
        # Get uploaded images
        face_frontal = request.FILES.get("face_frontal")
        face_left = request.FILES.get("face_left")
        face_right = request.FILES.get("face_right")
        use_for_training = request.POST.get("use_for_training", False)
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
            model_res_text = process_results_for_template(results)
            result = Result.objects.create(
                user = user,
                face_frontal = visualizations['face_frontal']['orignal_face'],
                face_left = visualizations['face_left']['orignal_face'],
                face_right = visualizations['face_right']['orignal_face'],
                yolo_face_front = results['face_frontal']['face'],
                yolo_face_left = results['face_frontal']['face'],
                yolo_face_right = results['face_frontal']['face'],
                general_disease_masks_front = visualizations['face_frontal']['general_diseases'],
                general_disease_masks_left= visualizations['face_left']['general_diseases'],
                general_disease_masks_right= visualizations['face_right']['general_diseases'],

                acne_mask_front = visualizations['face_frontal']['acne'],
                acne_mask_left = visualizations['face_left']['acne'],
                acne_mask_right = visualizations['face_right']['acne'],

                face_frontal_results = model_res_text['face_frontal'],
                face_left_results = model_res_text['face_left'],
                face_right_results = model_res_text['face_right'],

                use_for_training = use_for_training 

            )
            result.save()
            print(f"Result saved with ID: {result.id}")
            # This template will redirect to result page to avoid time consumption in development and to view old results
            return redirect('view_results', result.id)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing images: {str(e)}\n{traceback.format_exc()}"
            messages.error(request, error_msg)
            return render(request, 'predict.html', context={"error": error_msg})
    
    # GET request: show the upload form
    return render(request, 'predict.html',context={'angles':["frontal", "left", "right"]})



def view_results(request, result_id):
    try:
        result = Result.objects.get(id=result_id)
        
        return render(request, 'results.html', {
            'result': result,

        })
    except Result.DoesNotExist:
        messages.error(request, "Result not found.")
        return redirect('predict')

def overlay_mask_on_face(face_image, mask, color=(0, 255, 0), alpha=1.0):
    """
    Overlay a binary mask on a face image.

    Args:
        face_image (np.ndarray): Original face image (H x W x 3).
        mask (np.ndarray): Binary mask (H x W) or (H x W x 1).
        color (tuple): RGB color for the mask overlay.
        alpha (float): Transparency factor for the overlay.

    Returns:
        np.ndarray: Image with mask overlaid.
    """
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Create a color version of the mask
    colored_mask = np.zeros_like(face_image)
    colored_mask[mask > 0] = color  # Apply color where mask is non-zero

    # Blend the original image with the colored mask
    overlayed = cv2.addWeighted(colored_mask, alpha, face_image, 1 - alpha, 0)

    # Where mask is zero, keep the original image
    result = np.where(mask[:, :, np.newaxis] > 0, overlayed, face_image)

    return result

def generate_visualizations(results):
    """Generate visualizations for the results to display in the UI with overlaid masks and class names."""
    visualizations = {}
    CLASS_COLORS = {
        'Blackheads':      (255, 0, 0, 255),     # Red
        'Dark-Spots':      (255, 165, 0, 255),   # Orange
        'Dry-Skin':        (255, 255, 0, 255),   # Yellow
        'Englarged-Pores': (0, 128, 0, 255),     # Dark Green
        'Eyebags':         (0, 255, 255, 255),   # Cyan
        'Oily-Skin':       (0, 0, 255, 255),     # Blue
        'Skin-Redness':    (128, 0, 128, 255),   # Purple
        'Whiteheads':      (255, 192, 203, 255), # Pink
        'Wrinkles':        (139, 69, 19, 255),   # Brown
        'Acne':            (0, 255, 0, 255),     # Green
    }

    # Class names indexed from 0
    class_names = [
        'Acne', 'Blackheads', 'Dark-Spots', 'Dry-Skin', 'Englarged-Pores',
        'Eyebags', 'Oily-Skin', 'Skin-Redness', 'Whiteheads', 'Wrinkles'
    ]

    for face_direction, face_results in results.items():
        face_viz = {}

        # Original face image (for display)
        face_image = face_results["face_image"]
        face_viz["face"] = image_to_base64(face_image)
        face_viz['orignal_face'] = image_to_base64(face_results["original_image"])

        # General disease overlays
        general_binary_masks = face_results["general_disease_binary_masks"]
        general_viz = []

        for i, mask in enumerate(general_binary_masks):
            if i == 0:
                continue
            # print(f"Mask shape for class {i}: {mask.shape}")
            # print(f"face_image shape: {face_image.shape}")
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            # Overlay mask on face image
            mask_image = binary_mask_to_png( mask,color=CLASS_COLORS[class_name])
            # mask_image = image_to_base64(overlayed)
            general_viz.append({
                "class_name": class_name,
                "image": mask_image
            })

        face_viz["general_diseases"] = general_viz

        # Acne-specific overlay
        acne_mask = face_results["specific_acne_binary_mask"]
        acne_overlay = binary_mask_to_png(acne_mask,color=CLASS_COLORS['Acne'])
        face_viz["acne"] = acne_overlay

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

def binary_mask_to_png(mask, color=(0, 255, 0, 255)):
    """
    Convert a binary mask (numpy array with 0 and 1) to a base64 PNG image.
    Foreground (1) will be shown in the provided RGBA color, background (0) will be transparent.
    """
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[mask == 1] = color
    return image_to_base64(rgba_image)



def process_results_for_template(results):
    """Process the raw prediction results for template rendering"""
    processed_results = {}
    class_names = [
        'Acne', 'Blackheads', 'Dark-Spots', 'Dry-Skin', 'Englarged-Pores',
        'Eyebags', 'Oily-Skin', 'Skin-Redness', 'Whiteheads', 'Wrinkles'
    ]
    # Calculate metrics for each face direction
    for face_direction, face_results in results.items():
        # Calculate percentage of affected area for each disease class
        general_disease_percentages = []
        
        for i, mask in enumerate(face_results["general_disease_binary_masks"]):
            if i == 0:
                # Skip the first mask (assumed to be acne)
                continue

            percentage = (np.sum(mask > 0) / mask.size) * 100
            general_disease_percentages.append({
                "class_name": f"{class_names[i]}",
                "affected_percentage": round(percentage, 2)
            })
        
        # Calculate acne affected percentage
        acne_mask = face_results["specific_acne_binary_mask"]
        acne_percentage = (np.sum(acne_mask > 0) / acne_mask.size) * 100
        
        processed_results[face_direction] = {
            "general_diseases": general_disease_percentages,
            "acne": {"affected_percentage": round(acne_percentage, 2)}
        }
        print(f"Processed results for {face_direction}: {processed_results[face_direction]}")
    
    return processed_results




def llm_endpoint(request):
    # Write initial prompt add RAG capabilities Use gemini and or mistral with ollama running on another endpoint
    if request.method == 'POST':
        # Get the input text from the request
        input_text = request.POST.get('input_text', '')

        # Get Chat history from the request (if any)
        chat_history = request.POST.get('chat_history', '')

        if chat_history:
            chat_history = json.loads(chat_history)
        
        else:
            chat_history =[]
        
        
        # Process the input text using the LLM (this is a placeholder for actual LLM processing)
        # For now, we'll just echo back the input text
        response_text = f"Processed: {input_text}"
        
        # Return the response as JSON
        return JsonResponse({'response': response_text})
    
    # If not a POST request, return an empty response or render a template
    return JsonResponse({'response': ''})