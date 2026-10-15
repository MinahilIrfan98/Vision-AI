import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import random

def get_color(cls_id):
    """Returns a consistent BGR color for a given class ID."""
    # Fixed colors for primary objects
    if cls_id == 0: return (0, 255, 0) # Person Green
    if cls_id == 67: return (0, 0, 255) # Phone Red/Pink (BGR)

    # Random but deterministic colors for other 78 classes - shifted towards purple/blue
    random.seed(cls_id)
    return (random.randint(100, 255), random.randint(0, 100), random.randint(100, 255))

def is_inside(inner_box, outer_box):
    """Checks if inner_box is mostly inside outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    center_x = (ix1 + ix2) / 2
    center_y = (iy1 + iy2) / 2
    return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2

def draw_filled_label(img, box, text, color):
    """Draws a filled label tag on the bounding box."""
    x1, y1, x2, y2 = map(int, box)
    # Label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
    # Label text
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def detect_phone_usage(image, model, conf_threshold=0.35, hide_objects=None, show_conf=False):
    """
    Detects all objects, filters out hidden ones, marks phone usage,
    and returns annotated image and stats.
    """
    if hide_objects is None:
        hide_objects = []

    annotated_img = image.copy()
    results = model(image, conf=conf_threshold)[0]

    persons = []
    phones = []
    all_detected = {} # {class_name: count}

    # Filter detections
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        label = model.names[cls]

        if label in hide_objects:
            continue

        all_detected[label] = all_detected.get(label, 0) + 1

        if cls == 0:
            persons.append({'box': coords, 'conf': conf})
        elif cls == 67:
            phones.append({'box': coords, 'conf': conf})

    using_phone_count = 0

    # 1. Draw Phones first (so persons can overlap)
    for ph in phones:
        box = ph['box']
        color = get_color(67)
        txt = "Phone"
        if show_conf: txt += f" {ph['conf']:.2f}"
        cv2.rectangle(annotated_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        draw_filled_label(annotated_img, box, txt, color)

    # 2. Draw Persons
    for p in persons:
        p_box = p['box']
        is_using = False
        for ph in phones:
            if is_inside(ph['box'], p_box):
                is_using = True
                break

        if is_using:
            using_phone_count += 1
            color = (0, 0, 255) # Red
            label = "Using phone"
        else:
            color = (0, 255, 0) # Green
            label = "Person"

        if show_conf:
            label += f" {p['conf']:.2f}"

        cv2.rectangle(annotated_img, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), color, 2)
        draw_filled_label(annotated_img, p_box, label, color)

    # 3. Draw Other Objects
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in hide_objects or cls == 0 or cls == 67:
            continue

        coords = box.xyxy[0].tolist()
        color = get_color(cls)
        txt = label
        if show_conf:
            txt += f" {float(box.conf[0]):.2f}"
        cv2.rectangle(annotated_img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), color, 2)
        draw_filled_label(annotated_img, coords, txt, color)

    stats = {
        "persons": len(persons),
        "phones": len(phones),
        "using_phone": using_phone_count,
        "all_detected": all_detected
    }

    return annotated_img, stats

def main():
    model = YOLO('yolo26n.pt')
    input_dir = Path('images')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    if not image_files:
        print("No images found in the 'images/' folder.")
        return

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        annotated, _ = detect_phone_usage(img, model)
        cv2.imwrite(str(output_dir / img_path.name), annotated)

if __name__ == "__main__":
    main()
