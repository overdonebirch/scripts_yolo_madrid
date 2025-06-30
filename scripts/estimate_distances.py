#!/usr/bin/env python3
"""
estimate_distances.py
Estimate relative distance from camera to detected objects using MiDaS depth estimation.
"""
import os
import json
import argparse
import numpy as np
import torch
import cv2

# Map face indices to names
FACE_NAMES = ["front", "right", "back", "left", "up", "down"]

def load_midas(model_type="small", device="cpu"):
    """
    Load MiDaS model and corresponding transform.
    """
    if model_type == "small":
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    else:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
    model.to(device).eval()
    return model, transforms

def estimate_depth_for_face(model, transforms, device, face_image_path):
    """
    Estimate depth map for a single cubemap face image.
    """
    img = cv2.imread(face_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {face_image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transforms(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
    depth = prediction.squeeze().cpu().numpy()
    # Resize to original image size
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return depth

def main():
    parser = argparse.ArgumentParser(description="Estimate distances using MiDaS model.")
    parser.add_argument("-d", "--detections", default="cubemap_output/detections.json", help="Path to detections.json")
    parser.add_argument("-f", "--faces_dir", default="cubemap_output", help="Directory with cubemap face images")
    parser.add_argument("-o", "--output", default="cubemap_output/distances.json", help="Output JSON file for distances")
    parser.add_argument("--model_type", choices=["small", "large"], default="small", help="MiDaS model size")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Torch device")
    args = parser.parse_args()

    # Load detections
    if not os.path.exists(args.detections):
        print(f"Detections not found: {args.detections}")
        return
    with open(args.detections, 'r') as f:
        dets = json.load(f)

    # Load MiDaS model
    device = args.device
    model, transforms = load_midas(args.model_type, device)

    results = {}

    for face_str, data in dets.items():
        face = int(face_str)
        face_name = FACE_NAMES[face] if face < len(FACE_NAMES) else face_str
        face_img = os.path.join(args.faces_dir, f"{face_name}.jpg")
        # Depth map for this face
        depth_map = estimate_depth_for_face(model, transforms, device, face_img)
        boxes = data.get('boxes', [])
        scores = data.get('scores', [])
        classes = data.get('classes', [])
        face_results = []
        for idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox)
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                dist = None
            else:
                dist = float(np.median(region))
            face_results.append({
                'bbox_index': idx,
                'class_id': int(classes[idx]) if idx < len(classes) else None,
                'score': float(scores[idx]) if idx < len(scores) else None,
                'distance': dist
            })
        results[face_name] = face_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Estimated distances saved to {args.output}")

if __name__ == '__main__':
    main()
