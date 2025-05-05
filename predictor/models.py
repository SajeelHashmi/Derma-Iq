from django.db import models
from django.contrib.auth.models import User


class Result(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='results')
    created_at = models.DateTimeField(auto_now_add=True)
    use_data_for_training = models.BooleanField(default=False)

    # Base64-encoded images
    face_front = models.TextField()
    face_left = models.TextField()
    face_right = models.TextField()

    yolo_face_front = models.TextField()
    yolo_face_left = models.TextField()
    yolo_face_right = models.TextField()

    # General disease masks (base64 strings)
    general_disease_masks_front = models.JSONField()
    general_disease_masks_left = models.JSONField()
    general_disease_masks_right = models.JSONField()

    # Acne masks (base64 strings)
    acne_mask_front = models.TextField()
    acne_mask_left = models.TextField()
    acne_mask_right = models.TextField()

    face_frontal_results = models.JSONField()
    face_left_results = models.JSONField()
    face_right_results = models.JSONField()

    public = models.BooleanField(default=False)

    def __str__(self):
        return f"Result {self.id} for {self.user.username} on {self.created_at}"