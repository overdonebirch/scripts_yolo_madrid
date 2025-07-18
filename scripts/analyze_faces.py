#!/usr/bin/env python3
"""
analyze_faces.py
Script para ejecutar detección YOLO en imágenes de caras de cubemap generadas previamente.
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO detection on cubemap faces")
    parser.add_argument("-f", "--faces-dir", required=True, help="Directorio con caras del cubemap (.jpg)")
    parser.add_argument("-m", "--model", default="best.pt", help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-o", "--output-dir", default=None, help="Directorio de salida (por defecto faces-dir)")
    args = parser.parse_args()

    faces_dir = args.faces_dir
    model_path = args.model
    output_dir = args.output_dir or faces_dir
    os.makedirs(output_dir, exist_ok=True)

    face_names = ["front", "right", "back", "left", "up", "down"]
    colors = [
        (255, 0, 0),    # Rojo
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Azul
        (255, 255, 0),  # Amarillo
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cian
    ]

    print(f"Iniciando detección YOLO con modelo: {model_path}")
    model = YOLO(model_path)
    detections = {}

    for idx, face_name in enumerate(face_names):
        face_file = f"{face_name}.jpg"
        face_path = os.path.join(faces_dir, face_file)
        if not os.path.isfile(face_path):
            print(f"Warning: no se encontró {face_path}")
            detections[face_name] = {"boxes": [], "num_detections": 0}
            continue

        print(f"Procesando cara {idx}: {face_file}")
        results = model.predict(source=face_path, save=False, save_txt=False, verbose=False)
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
        else:
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])

        # Crear lista de boxes con scores y classes incluidos
        boxes_with_data = []
        for i in range(len(boxes)):
            box_data = {
                "coordinates": boxes[i].tolist(),
                "score": float(scores[i]),
                "class": int(classes[i])
            }
            boxes_with_data.append(box_data)
        
        detections[face_name] = {
            "boxes": boxes_with_data,
            "num_detections": int(len(boxes))
        }

        # Dibujar y guardar imagen con detecciones
        if len(boxes_with_data) > 0:
            img = Image.open(face_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            for box_data in boxes_with_data:
                x1, y1, x2, y2 = box_data["coordinates"]
                cls = box_data["class"]
                score = box_data["score"]
                color = colors[cls % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                text = f"{cls}: {score:.2f}"
                text_pos = (x1, max(0, y1 - 10))
                # Fondo para legibilidad
                draw.rectangle([text_pos, (text_pos[0] + len(text)*6, text_pos[1] + 12)], fill=(0, 0, 0, 128))
                draw.text(text_pos, text, fill=color, font=font)
            out_img = os.path.join(output_dir, f"{face_name}_with_detections.jpg")
            img.save(out_img, quality=95)
            print(f"  -> Guardado {out_img}")

    # Guardar JSON de detecciones
    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Detecciones guardadas en: {json_path}")

if __name__ == '__main__':
    main()
