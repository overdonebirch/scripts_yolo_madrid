import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os
import json


import argparse

class CubemapBBoxConverter:
    def __init__(self, input_image_path, output_dir="cubemap_output", cube_size=None):
        """
        Conversor de cubemap con soporte para bounding boxes
        
        Args:
            input_image_path: Ruta de la imagen 360° equirectangular
            output_dir: Directorio donde guardar las caras del cubo
            cube_size: Tamaño de cada cara del cubo
        """
        self.input_path = input_image_path
        self.output_dir = output_dir
        self.image = None
        self.width = 0
        self.height = 0
        self.cube_size = cube_size
        
        # Para almacenar detecciones de YOLO
        self.detections = {}  # face_index -> detections
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """Carga la imagen 360°"""
        try:
            self.image = Image.open(self.input_path)
            self.width, self.height = self.image.size
            
            if self.cube_size is None:
                self.cube_size = self.width // 4
                
            print(f"Imagen cargada: {self.width}x{self.height}")
            print(f"Tamaño de cara del cubo: {self.cube_size}x{self.cube_size}")
            
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
        return True
    
    def equirectangular_to_cubemap_coord(self, face, i, j):
        """Convierte coordenadas del cubo a coordenadas equirectangulares"""
        a = 2.0 * i / self.cube_size - 1.0
        b = 1.0 - 2.0 * j / self.cube_size
        
        if face == 0:  # Frente (+Z)
            x, y, z = a, b, 1.0
        elif face == 1:  # Derecha (+X)
            x, y, z = 1.0, b, -a
        elif face == 2:  # Atrás (-Z)
            x, y, z = -a, b, -1.0
        elif face == 3:  # Izquierda (-X)
            x, y, z = -1.0, b, a
        elif face == 4:  # Arriba (+Y)
            x, y, z = a, 1.0, -b
        elif face == 5:  # Abajo (-Y)
            x, y, z = a, -1.0, b
        
        theta = math.atan2(y, math.sqrt(x*x + z*z))
        phi = math.atan2(x, z)
        
        img_x = (phi / math.pi + 1.0) * 0.5 * self.width
        img_y = (0.5 - theta / math.pi) * self.height
        
        img_x = max(0, min(self.width - 1, img_x))
        img_y = max(0, min(self.height - 1, img_y))
        
        return int(img_x), int(img_y)
    
    def cubemap_to_equirectangular_coord(self, face, cube_x, cube_y):
        """
        Convierte coordenadas de una cara del cubo a coordenadas equirectangulares
        
        Args:
            face: Índice de la cara (0-5)
            cube_x, cube_y: Coordenadas en la cara del cubo
            
        Returns:
            eq_x, eq_y: Coordenadas en imagen equirectangular
        """
        return self.equirectangular_to_cubemap_coord(face, cube_x, cube_y)
    
    def extract_face(self, face_index):
        """Extrae una cara específica del cubo"""
        face_image = Image.new('RGB', (self.cube_size, self.cube_size))
        face_pixels = face_image.load()
        source_pixels = self.image.load()
        
        for j in range(self.cube_size):
            for i in range(self.cube_size):
                src_x, src_y = self.equirectangular_to_cubemap_coord(face_index, i, j)
                face_pixels[i, j] = source_pixels[src_x, src_y]
        
        return face_image
    
    def convert_to_cubemap(self):
        """Convierte la imagen 360° a las 6 caras del cubemap"""
        if not self.load_image():
            return False
        
        face_names = ["front", "right", "back", "left", "up", "down"]
        face_paths = []
        
        print("Iniciando conversión a cubemap...")
        
        for face_index in range(6):
            print(f"Procesando cara {face_index + 1}/6: {face_names[face_index]}")
            
            face_image = self.extract_face(face_index)
            filename = f"{face_names[face_index]}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            face_image.save(filepath, quality=95)
            face_paths.append(filepath)
            
            print(f"Guardado: {filepath}")
        
        print("¡Conversión completada!")
        return face_paths
    
    def transform_bbox_to_equirectangular(self, face_index, bbox):
        """
        Transforma un bounding box de una cara del cubo a coordenadas equirectangulares
        
        Args:
            face_index: Índice de la cara del cubo
            bbox: [x1, y1, x2, y2] en coordenadas de la cara del cubo
            
        Returns:
            Lista de puntos [(x, y)] que forman el contorno en coordenadas equirectangulares
        """
        x1, y1, x2, y2 = bbox
        
        # Crear puntos del perímetro del bounding box
        perimeter_points = []
        
        # Borde superior (y1)
        for x in range(int(x1), int(x2) + 1, max(1, int((x2-x1)/20))):
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x, y1)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde derecho (x2)
        for y in range(int(y1), int(y2) + 1, max(1, int((y2-y1)/20))):
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x2, y)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde inferior (y2)
        for x in range(int(x2), int(x1) - 1, -max(1, int((x2-x1)/20))):
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x, y2)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde izquierdo (x1)
        for y in range(int(y2), int(y1) - 1, -max(1, int((y2-y1)/20))):
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x1, y)
            perimeter_points.append((eq_x, eq_y))
        
        return perimeter_points


def main():
    """Función principal que ejecuta todo el pipeline"""
            
    # Configuración de parámetros
    parser = argparse.ArgumentParser(description="Pipeline: 360° → cubemap → YOLO → anotaciones")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen equirectangular 360°")
    parser.add_argument("-o", "--output-dir", default="cubemap_output", help="Directorio de salida")
    parser.add_argument("-c", "--cube-size", type=int, default=None, help="Tamaño de cada cara del cubemap")
    args = parser.parse_args()
    input_image = args.image
    output_directory = args.output_dir
    cube_size = args.cube_size
            
    print("=== Pipeline: 360° -> Cubemap ===\n")
            
    # Crear conversor
    converter = CubemapBBoxConverter(input_image, output_directory, cube_size)
            
    # Paso 1: Convertir a cubemap
    print("1. Convertiendo imagen 360° a cubemap...")
    face_paths = converter.convert_to_cubemap()
    if not face_paths:
        print("Error en la conversión a cubemap")
        return
    
    print(f"\n¡Conversión completada! Revisa la carpeta: {output_directory}")

if __name__ == "__main__":
    main()