#!/usr/bin/env python3
"""
run_pipeline.py
Procesa imágenes 360 en lote o individualmente:
  - Convierte a cubemap
  - Calcula azimuths
  - Estima distancias con UniDepth
  - Calcula coordenadas GPS
Estructura de proyecto:
  scripts/       (contiene los scripts .py)
  modelos/       (contiene modelo YOLO .pt)
  imagenes/      (contiene imágenes .jpg/.png)
  run_pipeline.py
"""
import os
import glob
import subprocess
import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Pipeline de detección geolocalizada para imágenes 360°")
    parser.add_argument('-i','--image', help="Procesar solo esta imagen (nombre o ruta)")
    parser.add_argument('-c','--cube-size', type=int, help="Tamaño de cara del cubemap")
    parser.add_argument('-m','--model', default=os.path.join('modelos','best.pt'),
                        help="Ruta al modelo YOLO (.pt)")
    parser.add_argument('--version', default='v2', help="Versión UniDepth (v1,v2,v2old)")
    parser.add_argument('--backbone', default='vitl14', help="Backbone UniDepth")
    parser.add_argument('--images-dir', default='imagenes', help="Directorio de imágenes 360°")
    parser.add_argument('--scripts-dir', default='scripts', help="Directorio de scripts Python")
    parser.add_argument('--output-root', default='outputs', help="Directorio raíz de salida (carpeta outputs)")
    parser.add_argument('--logs-dir', default='logs', help="Directorio raíz de logs (carpeta logs)")
    args = parser.parse_args()
    # Asegurar existencia de carpeta raíz de outputs
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # Listado de imágenes a procesar
    if args.image:
        img = args.image
        if not os.path.isfile(img):
            img = os.path.join(args.images_dir, args.image)
        if not os.path.isfile(img):
            logging.error(f"Imagen no encontrada: {args.image}")
            sys.exit(1)
        images = [img]
    else:
        pattern = os.path.join(args.images_dir, '*')
        images = [f for f in glob.glob(pattern)
                  if os.path.isfile(f) and f.lower().endswith(('.jpg','.jpeg','.png'))]
        images.sort()

    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        out_dir = os.path.join(args.output_root, f"output_{name}")
        os.makedirs(out_dir, exist_ok=True)
        # Abrir log de la imagen
        log_path = os.path.join(args.logs_dir, f"logs_{name}.txt")
        log_f = open(log_path, 'w')
        logging.info(f"Procesando {img} → {out_dir}")

        # 1. Cubemap
        cmd1 = [sys.executable, os.path.join(args.scripts_dir,'convert_images.py'),
                '-i', img, '-o', out_dir]
        if args.cube_size:
            cmd1 += ['-c', str(args.cube_size)]
        try:
            subprocess.run(cmd1, check=True, stdout=log_f, stderr=log_f)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error en convert_images.py: {e}")
            continue

                # 2. Detección YOLO en caras del cubemap
        cmd_detect = [sys.executable, os.path.join(args.scripts_dir,'analyze_faces.py'),
                '-f', out_dir, '-m', args.model]
        try:
            subprocess.run(cmd_detect, check=True, stdout=log_f, stderr=log_f)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error en analyze_faces.py: {e}")
            continue

        # 3. Azimuths
        det_json = os.path.join(out_dir, 'detections.json')
        az_out = os.path.join(out_dir, 'azimuths.json')
        cmd2 = [sys.executable, os.path.join(args.scripts_dir,'calculate_azimuths.py'),
                '-i', img, '-d', det_json, '-o', az_out]
        try:
            subprocess.run(cmd2, check=True, stdout=log_f, stderr=log_f)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error en calculate_azimuths.py: {e}")
            continue

                # 4. Distancias
        dist_out = os.path.join(out_dir, 'distances.json')
        cmd3 = [sys.executable, os.path.join(args.scripts_dir,'estimate_distances.py'),
                '-d', det_json, '-f', out_dir, '-o', dist_out,
                '--version', args.version, '--backbone', args.backbone]
        try:
            subprocess.run(cmd3, check=True, stdout=log_f, stderr=log_f)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error en estimate_distances.py: {e}")
            continue

                # 5. Coordenadas GPS
        coords_out = os.path.join(out_dir, 'coords.json')
        cmd4 = [sys.executable, os.path.join(args.scripts_dir,'compute_geo_coords.py'),
                '-i', img, '-a', az_out, '-d', dist_out, '-o', coords_out]
        try:
            subprocess.run(cmd4, check=True, stdout=log_f, stderr=log_f)
        except subprocess.CalledProcessError:
            logging.warning(f"Sin EXIF GPS en {img}, se omiten coords")
        except Exception as e:
            logging.error(f"Error en compute_geo_coords.py: {e}")
        log_f.close()

    logging.info("Pipeline finalizado.")

if __name__ == '__main__':
    main()
