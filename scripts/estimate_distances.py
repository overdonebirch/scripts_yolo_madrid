#!/usr/bin/env python3
"""
estimate_distances_unidepth.py
Estimate absolute distances (m) per detection using UniDepth.
"""
import os
import json
import argparse
import numpy as np
import torch
import cv2
from torchvision import transforms

# Names for cubemap faces
FACE_NAMES = ["front", "right", "back", "left", "up", "down"]

# Default transform: resize to model input and normalize imagenet
DEFAULT_SHAPE = (480, 640)  # height, width from UniDepth config
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(DEFAULT_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_unidepth(version, backbone, device):
    model = torch.hub.load(
        'lpiccinelli-eth/UniDepth', 'UniDepth',
        version=version, backbone=backbone, pretrained=True
    )
    model.to(device).eval()
    return model


def estimate_depth_face(model, device, img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = TRANSFORM(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        # use UniDepth's infer to get absolute depth in meters
        out = model.infer(inp)
    # out["depth"] tensor shape [B,1,H',W']
    depth_tensor = out["depth"]
    depth = depth_tensor.squeeze().cpu().numpy()
    # resize back to original resolution
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return depth


def main():
    p = argparse.ArgumentParser(description="Estimate distances with UniDepth (absolute in meters)")
    p.add_argument("-d","--detections", default="cubemap_output/detections.json")
    p.add_argument("-f","--faces_dir", default="cubemap_output")
    p.add_argument("-o","--output", default="cubemap_output/distances_unidepth.json")
    p.add_argument("--version", choices=["v1","v2","v2old"], default="v2")
    p.add_argument("--backbone", default="vitl14")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    if not os.path.exists(args.detections):
        print(f"Detections not found: {args.detections}")
        return
    dets = json.load(open(args.detections))

    device = args.device
    model = load_unidepth(args.version, args.backbone, device)

    results = {}
    for face_str, data in dets.items():
        face = int(face_str)
        name = FACE_NAMES[face] if face < len(FACE_NAMES) else face_str
        img_path = os.path.join(args.faces_dir, f"{name}.jpg")
        depth_map = estimate_depth_face(model, device, img_path)
        face_out = []
        for idx, bbox in enumerate(data.get('boxes', [])):
            x1,y1,x2,y2 = map(int, bbox)
            region = depth_map[y1:y2, x1:x2]
            dist = float(np.median(region)) if region.size>0 else None
            face_out.append({
                'bbox_index': idx,
                'class_id': int(data.get('classes', [])[idx]) if idx < len(data.get('classes', [])) else None,
                'score': float(data.get('scores', [])[idx]) if idx < len(data.get('scores', [])) else None,
                'distance_m': dist
            })
        results[name] = face_out

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output,'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved distances (m) to {args.output}")

if __name__=='__main__':
    main()
