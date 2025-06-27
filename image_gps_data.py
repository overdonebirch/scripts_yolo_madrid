#!/usr/bin/env python3
"""
Script para extraer metadatos de imágenes 360 y verificar datos de geolocalización
Soporta formatos JPEG, TIFF y otros formatos con metadatos EXIF
"""

import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
from datetime import datetime

def dms_to_decimal(dms, ref):
    """
    Convierte coordenadas DMS (Degrees, Minutes, Seconds) a decimal
    """
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    decimal = degrees + minutes + seconds
    
    if ref in ['S', 'W']:
        decimal = -decimal
    
    return decimal

def extract_gps_data(exif_data):
    """
    Extrae datos GPS de los metadatos EXIF
    """
    gps_data = {}
    
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        
        # Procesar cada tag GPS
        for key in gps_info.keys():
            name = GPSTAGS.get(key, key)
            gps_data[name] = gps_info[key]
        
        # Extraer coordenadas si están disponibles
        if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
            lat = dms_to_decimal(gps_data['GPSLatitude'], gps_data['GPSLatitudeRef'])
            gps_data['DecimalLatitude'] = lat
            
        if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
            lon = dms_to_decimal(gps_data['GPSLongitude'], gps_data['GPSLongitudeRef'])
            gps_data['DecimalLongitude'] = lon
            
        # Extraer altitud si está disponible
        if 'GPSAltitude' in gps_data:
            altitude = float(gps_data['GPSAltitude'])
            altitude_ref = gps_data.get('GPSAltitudeRef', 0)
            if altitude_ref == 1:  # Por debajo del nivel del mar
                altitude = -altitude
            gps_data['DecimalAltitude'] = altitude
    
    return gps_data

def extract_360_metadata(image_path):
    """
    Extrae metadatos específicos de imágenes 360
    """
    metadata_360 = {}
    
    try:
        with Image.open(image_path) as img:
            # Verificar si es una imagen 360 por sus metadatos XMP o dimensiones
            if hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif:
                    # Buscar metadatos específicos de 360
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if 'ProjectionType' in str(tag) or 'spherical' in str(value).lower():
                            metadata_360['Is360Image'] = True
                        if 'UsePanoramaViewer' in str(tag):
                            metadata_360['UsePanoramaViewer'] = value
            
            # Verificar dimensiones típicas de imágenes 360 (relación 2:1)
            width, height = img.size
            aspect_ratio = width / height
            if abs(aspect_ratio - 2.0) < 0.1:  # Tolerancia para relación 2:1
                metadata_360['PossibleEquirectangular'] = True
                metadata_360['AspectRatio'] = aspect_ratio
            
            metadata_360['ImageSize'] = f"{width}x{height}"
            
    except Exception as e:
        print(f"Error al procesar metadatos 360: {e}")
    
    return metadata_360

def extract_all_metadata(image_path):
    """
    Extrae todos los metadatos disponibles de la imagen
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no existe: {image_path}")
    
    metadata = {
        'file_path': image_path,
        'file_size': os.path.getsize(image_path),
        'extraction_time': datetime.now().isoformat(),
        'has_gps_data': False,
        'gps_coordinates': None,
        'exif_data': {},
        'metadata_360': {},
        'gps_data': {}
    }
    
    try:
        with Image.open(image_path) as img:
            # Información básica de la imagen
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['size'] = img.size
            
            # Extraer metadatos EXIF
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    # Convertir valores no serializables a string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    metadata['exif_data'][tag] = value
                
                # Extraer datos GPS
                gps_data = extract_gps_data(metadata['exif_data'])
                if gps_data:
                    metadata['gps_data'] = gps_data
                    metadata['has_gps_data'] = True
                    
                    # Crear resumen de coordenadas
                    if 'DecimalLatitude' in gps_data and 'DecimalLongitude' in gps_data:
                        metadata['gps_coordinates'] = {
                            'latitude': gps_data['DecimalLatitude'],
                            'longitude': gps_data['DecimalLongitude'],
                            'altitude': gps_data.get('DecimalAltitude', None)
                        }
            
            # Extraer metadatos específicos de 360
            metadata['metadata_360'] = extract_360_metadata(image_path)
            
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None
    
    return metadata

def print_metadata_summary(metadata):
    """
    Imprime un resumen de los metadatos extraídos
    """
    print("=" * 60)
    print("RESUMEN DE METADATOS DE IMAGEN 360")
    print("=" * 60)
    print(f"Archivo: {metadata['file_path']}")
    print(f"Tamaño: {metadata['file_size']} bytes")
    print(f"Formato: {metadata['format']}")
    print(f"Dimensiones: {metadata['size'][0]}x{metadata['size'][1]}")
    
    # Información sobre imagen 360
    print("\n--- INFORMACIÓN 360 ---")
    if metadata['metadata_360']:
        for key, value in metadata['metadata_360'].items():
            print(f"{key}: {value}")
    else:
        print("No se detectaron metadatos específicos de imagen 360")
    
    # Información GPS
    print("\n--- INFORMACIÓN DE GEOLOCALIZACIÓN ---")
    if metadata['has_gps_data']:
        print("✅ La imagen CONTIENE datos de geolocalización")
        if metadata['gps_coordinates']:
            coords = metadata['gps_coordinates']
            print(f"Latitud: {coords['latitude']:.6f}")
            print(f"Longitud: {coords['longitude']:.6f}")
            if coords['altitude']:
                print(f"Altitud: {coords['altitude']:.2f} metros")
            
            # Generar enlace a Google Maps
            lat, lon = coords['latitude'], coords['longitude']
            maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            print(f"Ver en Google Maps: {maps_url}")
    else:
        print("❌ La imagen NO contiene datos de geolocalización")
    
    # Otros metadatos EXIF relevantes
    print("\n--- OTROS METADATOS RELEVANTES ---")
    relevant_tags = ['DateTime', 'Make', 'Model', 'Software', 'ImageDescription']
    for tag in relevant_tags:
        if tag in metadata['exif_data']:
            print(f"{tag}: {metadata['exif_data'][tag]}")

def save_metadata_json(metadata, output_path):
    """
    Guarda los metadatos completos en un archivo JSON
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nMetadatos completos guardados en: {output_path}")
    except Exception as e:
        print(f"Error al guardar JSON: {e}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python script.py <ruta_imagen> [--json output.json]")
        print("Ejemplo: python script.py imagen360.jpg --json metadatos.json")
        return
    
    image_path = sys.argv[1]
    
    # Verificar si se solicita salida JSON
    save_json = False
    json_output = None
    if '--json' in sys.argv:
        json_index = sys.argv.index('--json')
        if json_index + 1 < len(sys.argv):
            json_output = sys.argv[json_index + 1]
            save_json = True
        else:
            json_output = 'metadata.json'
            save_json = True
    
    try:
        # Extraer metadatos
        metadata = extract_all_metadata(image_path)
        if metadata:
            # Mostrar resumen
            print_metadata_summary(metadata)
            
            # Guardar JSON si se solicita
            if save_json:
                save_metadata_json(metadata, json_output)
        else:
            print("No se pudieron extraer los metadatos de la imagen")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()