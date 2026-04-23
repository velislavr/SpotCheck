import os

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
from tensorflow.keras.models import load_model

from .gradcam import (
    make_gradcam_heatmap,
    overlay_heatmap,
    pil_to_base64_jpeg,
    preprocess_pil,
)

MODEL_PATH = os.path.join(settings.BASE_DIR, "skin_cancer_model.keras")
model = load_model(MODEL_PATH)


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PIXEL_DIM = 10_000


def predict(request):
    if request.method == "POST" and "file" in request.FILES:
        uploaded = request.FILES["file"]

        if uploaded.size > MAX_UPLOAD_SIZE:
            return JsonResponse({"error": "File too large (max 10 MB)"}, status=400)

        try:
            pil_image = Image.open(uploaded)
            pil_image.verify()
            uploaded.seek(0)
            pil_image = Image.open(uploaded)
        except Exception:
            return JsonResponse({"error": "Invalid image file"}, status=400)

        if max(pil_image.size) > MAX_PIXEL_DIM:
            return JsonResponse({"error": "Image dimensions too large"}, status=400)

        resized, img_array = preprocess_pil(pil_image)

        heatmap, malignant_prob = make_gradcam_heatmap(img_array, model)
        label = "Malignant" if malignant_prob >= 0.5 else "Benign"
        confidence = malignant_prob if label == "Malignant" else 1 - malignant_prob
        overlay = overlay_heatmap(resized, heatmap)

        return JsonResponse({
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "image": pil_to_base64_jpeg(resized),
            "heatmap": pil_to_base64_jpeg(overlay),
        })

    return JsonResponse({"error": "Invalid request"}, status=400)


def upload_view(request):
    return render(request, "predictions/upload.html")
