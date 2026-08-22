"""
Creates realistic synthetic brain MRI test images for the 4 classes
to allow instant testing and verification of the system.
"""

import os
import cv2
import numpy as np


def create_brain_mri_base(h=240, w=240):
    # Black background
    img = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2

    # Outer skull ellipse
    y, x = np.ogrid[:h, :w]
    skull_mask = (((x - cx) / (w * 0.40)) ** 2 + ((y - cy) / (h * 0.45)) ** 2) <= 1.0
    img[skull_mask] = 180.0

    # Inner brain parenchyma ellipse
    brain_mask = (((x - cx) / (w * 0.36)) ** 2 + ((y - cy) / (h * 0.41)) ** 2) <= 1.0
    img[brain_mask] = 100.0

    # Ventricles (butterfly shaped central ventricles)
    v1_mask = (((x - (cx - 15)) / 10.0) ** 2 + ((y - cy) / 28.0) ** 2) <= 1.0
    v2_mask = (((x - (cx + 15)) / 10.0) ** 2 + ((y - cy) / 28.0) ** 2) <= 1.0
    img[v1_mask | v2_mask] = 30.0

    # Brain folds / sulci texture
    noise = np.random.normal(0, 12, (h, w))
    img = np.clip(img + noise * brain_mask, 0, 255)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_samples(output_dir="sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. No Tumor
    normal_img = create_brain_mri_base(240, 240)
    cv2.imwrite(os.path.join(output_dir, "sample_notumor.png"), normal_img.astype(np.uint8))

    # 2. Glioma (Hyperintense irregular mass in right hemisphere)
    glioma_img = create_brain_mri_base(240, 240)
    y, x = np.ogrid[:240, :240]
    g_mask = (((x - 160) / 24.0) ** 2 + ((y - 95) / 20.0) ** 2) <= 1.0
    glioma_img[g_mask] = np.clip(glioma_img[g_mask] + 120, 0, 255)
    # Surrounding edema
    edema_mask = (((x - 160) / 32.0) ** 2 + ((y - 95) / 28.0) ** 2) <= 1.0
    glioma_img[edema_mask & ~g_mask] = np.clip(glioma_img[edema_mask & ~g_mask] + 50, 0, 255)
    glioma_img = cv2.GaussianBlur(glioma_img, (3, 3), 0)
    cv2.imwrite(os.path.join(output_dir, "sample_glioma.png"), glioma_img.astype(np.uint8))

    # 3. Meningioma (Dural-based extra-axial mass near skull convex)
    men_img = create_brain_mri_base(240, 240)
    m_mask = (((x - 65) / 18.0) ** 2 + ((y - 120) / 22.0) ** 2) <= 1.0
    men_img[m_mask] = np.clip(men_img[m_mask] + 135, 0, 255)
    men_img = cv2.GaussianBlur(men_img, (3, 3), 0)
    cv2.imwrite(os.path.join(output_dir, "sample_meningioma.png"), men_img.astype(np.uint8))

    # 4. Pituitary (Sellar / Parasellar mass at the skull base)
    pit_img = create_brain_mri_base(240, 240)
    p_mask = (((x - 120) / 16.0) ** 2 + ((y - 180) / 14.0) ** 2) <= 1.0
    pit_img[p_mask] = np.clip(pit_img[p_mask] + 140, 0, 255)
    pit_img = cv2.GaussianBlur(pit_img, (3, 3), 0)
    cv2.imwrite(os.path.join(output_dir, "sample_pituitary.png"), pit_img.astype(np.uint8))

    print(f"Generated 4 sample MRI images in {output_dir}")


if __name__ == "__main__":
    generate_samples()
