import numpy as np
from PIL import Image, ImageDraw
import math
import os
import json
from ultralytics import YOLO
import cv2

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
    
    def run_yolo_detection(self, model_path, face_paths):
        """
        Ejecuta detección YOLO en cada cara del cubemap
        
        Args:
            model_path: Ruta al modelo YOLO
            face_paths: Lista de rutas a las caras del cubemap
        """
        print("Iniciando detección YOLO en caras del cubemap...")
        
        model = YOLO(model_path)
        
        for face_index, face_path in enumerate(face_paths):
            print(f"Detectando en cara {face_index}: {face_path}")
            
            # Ejecutar detección
            results = model.predict(
                source=face_path,
                save=False,
                save_txt=False,
                verbose=False
            )
            
            # Guardar detecciones para esta cara
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # formato [x1, y1, x2, y2]
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                # También obtener máscaras si están disponibles
                masks = None
                if hasattr(results[0], 'masks') and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                
                self.detections[face_index] = {
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'masks': masks
                }
                
                print(f"  - Encontradas {len(boxes)} detecciones")
            else:
                self.detections[face_index] = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([]),
                    'masks': None
                }
                print(f"  - No se encontraron detecciones")
    
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
    
    def create_equirectangular_with_detections(self, output_path="output_with_detections.jpg"):
        """
        Crea imagen equirectangular original con bounding boxes superpuestos
        
        Args:
            output_path: Ruta donde guardar la imagen final
        """
        if not self.image:
            print("Error: Imagen no cargada")
            return False
        
        print("Creando imagen equirectangular con detecciones...")
        
        # Crear copia de la imagen original
        result_image = self.image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Colores para diferentes clases
        colors = [
            (255, 0, 0),    # Rojo
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarillo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cian
        ]
        
        total_detections = 0
        
        # Procesar detecciones de cada cara
        for face_index, detection_data in self.detections.items():
            if len(detection_data['boxes']) == 0:
                continue
                
            face_names = ["front", "right", "back", "left", "up", "down"]
            print(f"Procesando detecciones de cara {face_names[face_index]}...")
            
            for i, bbox in enumerate(detection_data['boxes']):
                # Transformar bounding box a coordenadas equirectangulares
                eq_points = self.transform_bbox_to_equirectangular(face_index, bbox)
                
                if len(eq_points) > 2:
                    # Elegir color basado en la clase
                    class_id = int(detection_data['classes'][i]) if len(detection_data['classes']) > i else 0
                    color = colors[class_id % len(colors)]
                    
                    # Dibujar el contorno
                    draw.polygon(eq_points, outline=color, width=3)
                    
                    # Agregar texto con la confianza
                    if len(detection_data['scores']) > i:
                        score = detection_data['scores'][i]
                        # Encontrar el punto más alto para colocar el texto
                        min_y_point = min(eq_points, key=lambda p: p[1])
                        draw.text((min_y_point[0], min_y_point[1] - 20), 
                                f"Tree: {score:.2f}", 
                                fill=color)
                    
                    total_detections += 1
        
        # Guardar imagen resultado
        output_full_path = os.path.join(self.output_dir, output_path)
        result_image.save(output_full_path, quality=95)
        
        print(f"Imagen con {total_detections} detecciones guardada en: {output_full_path}")
        return True
    
    def save_detections_json(self, output_path="detections.json"):
        """Guarda las detecciones en formato JSON"""
        detections_serializable = {}
        
        for face_index, detection_data in self.detections.items():
            detections_serializable[face_index] = {
                'boxes': detection_data['boxes'].tolist() if detection_data['boxes'].size > 0 else [],
                'scores': detection_data['scores'].tolist() if detection_data['scores'].size > 0 else [],
                'classes': detection_data['classes'].tolist() if detection_data['classes'].size > 0 else [],
                'num_detections': len(detection_data['boxes'])
            }
        
        json_path = os.path.join(self.output_dir, output_path)
        with open(json_path, 'w') as f:
            json.dump(detections_serializable, f, indent=2)
        
        print(f"Detecciones guardadas en: {json_path}")


def main():
    """Función principal que ejecuta todo el pipeline"""
    
    # Configuración
    input_image = "semaforo_dic2023.jpg"  # Tu imagen 360°
    model_path = "best.pt"  # Tu modelo YOLO
    output_directory = "cubemap_output"
    
    print("=== Pipeline Completo: 360° → Cubemap → YOLO → Restauración ===\n")
    
    # Crear conversor
    converter = CubemapBBoxConverter(input_image, output_directory)
    
    # Paso 1: Convertir a cubemap
    print("1. Convertiendo imagen 360° a cubemap...")
    face_paths = converter.convert_to_cubemap()
    if not face_paths:
        print("Error en la conversión a cubemap")
        return
    
    # Paso 2: Ejecutar YOLO en cada cara
    print("\n2. Ejecutando detección YOLO en cada cara...")
    converter.run_yolo_detection(model_path, face_paths)
    
    # Paso 3: Restaurar a formato equirectangular con detecciones
    print("\n3. Restaurando imagen equirectangular con detecciones...")
    converter.create_equirectangular_with_detections("resultado_final_con_detecciones.jpg")
    
    # Paso 4: Guardar detecciones en JSON
    print("\n4. Guardando detecciones en JSON...")
    converter.save_detections_json()
    
    print(f"\n¡Pipeline completado! Revisa la carpeta: {output_directory}")
    print("Archivos generados:")
    print("- 6 caras del cubemap (front.jpg, right.jpg, etc.)")
    print("- resultado_final_con_detecciones.jpg (imagen 360° con bounding boxes)")
    print("- detections.json (datos de todas las detecciones)")

if __name__ == "__main__":
    main()