import os
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image
from .ai_models import AIWorker
from datetime import datetime
import cv2 

def predict(request):
    if request.method == 'POST':
        face_frontal = request.FILES.get("face_frontal")
        face_left = request.FILES.get("face_left")
        face_right = request.FILES.get("face_right")
        faces = [face_frontal, face_left, face_right]

        if not all(faces):
            return render(request, 'predict.html', context={"error": "Please upload all three images."})

        ai_pipe = AIWorker()
        results = {}
        # open image and convert to numpy array
        face_frontal = Image.open(face_frontal)
        face_frontal = np.array(face_frontal)
        
        face_left = Image.open(face_left)
        face_right = np.array(face_right)
        
        face_right = Image.open(face_right)
        face_left = np.array(face_left)

        results['face_frontal'] = ai_pipe.predict(face_frontal)
        results['face_left'] = ai_pipe.predict(face_left)
        results['face_right'] = ai_pipe.predict(face_right)



    else:
        return render(request, 'predict.html',context={"angles":["front","left","right"]})
