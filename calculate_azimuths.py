#!/usr/bin/env python3
import json
import argparse
from math import atan2, degrees
from PIL import Image

# Map face indices to names
FACE_NAMES = ["front", "right", "back", "left", "up", "down"]

def compute_direction(face, i, j, cube_size):
    a = 2.0 * i / cube_size - 1.0
    b = 1.0 - 2.0 * j / cube_size
    if face == 0:  # front (+Z)
        x, y, z = a, b, 1.0
    elif face == 1:  # right (+X)
        x, y, z = 1.0, b, -a
    elif face == 2:  # back (-Z)
        x, y, z = -a, b, -1.0
    elif face == 3:  # left (-X)
        x, y, z = -1.0, b, a
    elif face == 4:  # up (+Y)
        x, y, z = a, 1.0, -b
    elif face == 5:  # down (-Y)
        x, y, z = a, -1.0, b
    else:
        raise ValueError(f"Invalid face index: {face}")
    return x, y, z


def compute_azimuth(face, i, j, cube_size):
    x, y, z = compute_direction(face, i, j, cube_size)
    phi = atan2(x, z)  # azimuth in radians
    deg = degrees(phi)
    if deg < 0:
        deg += 360.0
    return deg


def main():
    parser = argparse.ArgumentParser(description="Compute azimuths of YOLO detections in a cubemap.")
    parser.add_argument("-i", "--image", required=True, help="Path to the equirectangular 360° image.")
    parser.add_argument("-d", "--detections", required=True, help="Path to detections.json.")
    parser.add_argument("-o", "--output", help="Output JSON file for azimuths.", default=None)
    args = parser.parse_args()

    # Load image to get dimensions and cube size
    img = Image.open(args.image)
    width, height = img.size
    cube_size = width // 4

    # Load detections
    with open(args.detections, 'r') as f:
        dets = json.load(f)

    result = {}
    for face_str, data in dets.items():
        face = int(face_str)
        face_name = FACE_NAMES[face]
        boxes = data.get('boxes', [])
        classes = data.get('classes', [])
        az_list = []
        for idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            az = compute_azimuth(face, cx, cy, cube_size)
            az_list.append({
                'bbox_index': idx,
                'class_id': classes[idx] if idx < len(classes) else None,
                'azimuth_deg': az
            })
        result[face_name] = az_list

    output_str = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Azimuths saved to {args.output}")
    else:
        print(output_str)

if __name__ == '__main__':
    main()
