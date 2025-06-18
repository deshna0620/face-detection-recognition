import cv2
import os

def apply_mask(img_path, save_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    mask_color = (192, 192, 192)
    mask_height = int(h * 0.25)
    y1 = h - mask_height
    y2 = h
    cv2.rectangle(img, (0, y1), (w, y2), mask_color, -1)
    cv2.imwrite(save_path, img)

def apply_cap(img_path, save_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    cap_color = (50, 50, 50)
    cap_height = int(h * 0.25)
    y1 = 0
    y2 = cap_height
    cv2.rectangle(img, (0, y1), (w, y2), cap_color, -1)
    cv2.imwrite(save_path, img)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    apply_mask("data/face1.png", "data/face1_masked.png")
    apply_cap("data/face2.png", "data/face2_capped.png")
    print("Augmented mask/cap images saved.")