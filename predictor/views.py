import os
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image
from .unet import model  # Import the model once
from datetime import datetime
import cv2 

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')

        if uploaded_file:
            # Create timestamp for unique folder
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_path = os.path.join(settings.BASE_DIR, 'predictions')
            os.makedirs(base_path, exist_ok=True)

            # Save original image
            original_image_path = os.path.join(base_path, 'received.jpg')
            with open(original_image_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # Preprocess: grayscale, resize to 256x256x1
            img = Image.open(original_image_path).convert('L')  # Grayscale
            img = np.array(img)  # Convert to NumPy array

            # Apply histogram equalization using OpenCV
            equalized = cv2.equalizeHist(img)

            # Resize to 256x256
            equalized_resized = cv2.resize(equalized, (256, 256))

            # Normalize and reshape for model input
            img_array = equalized_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
            img_array = img_array[..., np.newaxis]  # Add channel dimension
            img_array = np.expand_dims(img_array, axis=0)
            # Predict with model → (1, 256, 256, 14)
            predictions = model.predict(img_array)[0]  # shape: (256, 256, 14)

            for mask_index in range(predictions.shape[-1]):
                mask_folder = os.path.join(base_path, f'mask_{mask_index}')
                os.makedirs(mask_folder, exist_ok=True)
                mask = predictions[..., mask_index]

                for threshold in np.arange(0.1, 1.0, 0.1):
                    binary_mask = (mask > threshold).astype(np.uint8) * 255
                    mask_img = Image.fromarray(binary_mask)
                    mask_img_path = os.path.join(mask_folder, f'thresh{threshold:.1f}.png')
                    mask_img.save(mask_img_path)

            return render(request, 'predict_success.html', {
                'path': base_path
            })
    else:
        return render(request, 'predict.html',context={"angles":["front","left","right"]})
