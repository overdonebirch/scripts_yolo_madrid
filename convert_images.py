import numpy as np
from PIL import Image
import math
import os

class CubemapConverter:
    def __init__(self, input_image_path, output_dir="cubemap_output", cube_size=None):
        """
        Inicializa el conversor de cubemap
        
        Args:
            input_image_path: Ruta de la imagen 360° equirectangular
            output_dir: Directorio donde guardar las caras del cubo
            cube_size: Tamaño de cada cara del cubo (si None, se calcula automáticamente)
        """
        self.input_path = input_image_path
        self.output_dir = output_dir
        self.image = None
        self.width = 0
        self.height = 0
        self.cube_size = cube_size
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """Carga la imagen 360°"""
        try:
            self.image = Image.open(self.input_path)
            self.width, self.height = self.image.size
            
            # Si no se especifica cube_size, usar 1/4 del ancho de la imagen
            if self.cube_size is None:
                self.cube_size = self.width // 4
                
            print(f"Imagen cargada: {self.width}x{self.height}")
            print(f"Tamaño de cara del cubo: {self.cube_size}x{self.cube_size}")
            
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
        return True
    
    def equirectangular_to_cubemap_coord(self, face, i, j):
        """
        Convierte coordenadas del cubo a coordenadas esféricas equirectangulares
        
        Args:
            face: Cara del cubo (0-5)
            i, j: Coordenadas en la cara del cubo
            
        Returns:
            x, y: Coordenadas en la imagen equirectangular
        """
        # Normalizar coordenadas de la cara del cubo a [-1, 1]
        a = 2.0 * i / self.cube_size - 1.0
        b = 1.0 - 2.0 * j / self.cube_size  # invertimos el eje vertical

        
        # Vectores 3D para cada cara del cubo (corregidos para orientación correcta)
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

        
        # Convertir a coordenadas esféricas
        theta = math.atan2(y, math.sqrt(x*x + z*z))  # Latitud
        phi = math.atan2(x, z)  # Longitud
        
        # Convertir a coordenadas de imagen equirectangular
        img_x = (phi / math.pi + 1.0) * 0.5 * self.width
        img_y = (0.5 - theta / math.pi) * self.height  # Invertir Y para corregir orientación
        
        # Asegurar que las coordenadas estén dentro de los límites
        img_x = max(0, min(self.width - 1, img_x))
        img_y = max(0, min(self.height - 1, img_y))
        
        return int(img_x), int(img_y)
    
    def extract_face(self, face_index):
        """
        Extrae una cara específica del cubo
        
        Args:
            face_index: Índice de la cara (0-5)
            
        Returns:
            PIL Image de la cara extraída
        """
        face_image = Image.new('RGB', (self.cube_size, self.cube_size))
        face_pixels = face_image.load()
        source_pixels = self.image.load()
        
        for j in range(self.cube_size):
            for i in range(self.cube_size):
                # Obtener coordenadas en la imagen original
                src_x, src_y = self.equirectangular_to_cubemap_coord(face_index, i, j)
                
                # Copiar pixel
                face_pixels[i, j] = source_pixels[src_x, src_y]
        
        return face_image
    
    def convert_to_cubemap(self):
        """Convierte la imagen 360° a las 6 caras del cubemap"""
        if not self.load_image():
            return False
        
        # Nombres de las caras
        face_names = [
            "front",    # +Z
            "right",    # +X  
            "back",     # -Z
            "left",     # -X
            "up",       # +Y
            "down"      # -Y
        ]
        
        print("Iniciando conversión a cubemap...")
        
        for face_index in range(6):
            print(f"Procesando cara {face_index + 1}/6: {face_names[face_index]}")
            
            # Extraer la cara
            face_image = self.extract_face(face_index)
            
            # Guardar la cara
            filename = f"{face_names[face_index]}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            face_image.save(filepath, quality=95)
            
            print(f"Guardado: {filepath}")
        
        print("¡Conversión completada!")
        return True
    
    def create_cross_layout(self):
        """Crea un layout en cruz con todas las caras del cubemap"""
        if not self.load_image():
            return False
        
        # Crear imagen en formato cruz (4x3 caras)
        cross_width = self.cube_size * 4
        cross_height = self.cube_size * 3
        cross_image = Image.new('RGB', (cross_width, cross_height), (0, 0, 0))
        
        # Layout en cruz:
        #     [up]
        # [left][front][right][back]
        #     [down]
        
        face_positions = [
            (1, 1),  # front
            (2, 1),  # right
            (3, 1),  # back
            (0, 1),  # left
            (1, 0),  # up
            (1, 2),  # down
        ]
        
        print("Creando layout en cruz...")
        
        for face_index in range(6):
            face_image = self.extract_face(face_index)
            pos_x, pos_y = face_positions[face_index]
            
            # Pegar la cara en la posición correcta
            cross_image.paste(face_image, 
                            (pos_x * self.cube_size, pos_y * self.cube_size))
        
        # Guardar layout en cruz
        cross_path = os.path.join(self.output_dir, "cubemap_cross.jpg")
        cross_image.save(cross_path, quality=95)
        print(f"Layout en cruz guardado: {cross_path}")
        
        return True

def main():
    """Función principal de ejemplo"""
    # Configuración
    input_image = "imagen_sigo.jpg"  # Cambia por la ruta de tu imagen
    output_directory = "cubemap_output"
    
    # Crear conversor
    converter = CubemapConverter(input_image, output_directory)
    
    print("=== Conversor de Imagen 360° a Cubemap ===\n")
    
    # Opción 1: Generar caras individuales
    print("1. Generando caras individuales del cubemap...")
    if converter.convert_to_cubemap():
        print("✓ Caras individuales generadas correctamente\n")
    else:
        print("✗ Error al generar las caras individuales\n")
        return
    
    # Opción 2: Generar layout en cruz
    print("2. Generando layout en cruz...")
    if converter.create_cross_layout():
        print("✓ Layout en cruz generado correctamente\n")
    else:
        print("✗ Error al generar el layout en cruz\n")
    
    print("Proceso completado. Revisa la carpeta:", output_directory)

if __name__ == "__main__":
    main()