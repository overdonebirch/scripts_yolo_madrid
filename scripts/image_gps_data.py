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
    Extrae datos GPS de los metadatos EXIF incluyendo información detallada de altitud
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
            
        # Extraer altitud GPS si está disponible
        if 'GPSAltitude' in gps_data:
            altitude = float(gps_data['GPSAltitude'])
            altitude_ref = gps_data.get('GPSAltitudeRef', 0)
            if altitude_ref == 1:  # Por debajo del nivel del mar
                altitude = -altitude
            gps_data['DecimalAltitude'] = altitude
            gps_data['AltitudeSource'] = 'GPS'
        
        # Información adicional de altitud y altura
        altitude_info = {}
        
        # Diferencial geodésico (altura sobre el geoide)
        if 'GPSHPositioningError' in gps_data:
            altitude_info['HorizontalAccuracy'] = gps_data['GPSHPositioningError']
        
        # Información de DOP (Dilution of Precision)
        if 'GPSDOP' in gps_data:
            altitude_info['DOP'] = gps_data['GPSDOP']
        
        # Método de medición GPS
        if 'GPSMeasureMode' in gps_data:
            mode = gps_data['GPSMeasureMode']
            if mode == '2':
                altitude_info['GPSMode'] = '2D (sin altitud)'
            elif mode == '3':
                altitude_info['GPSMode'] = '3D (con altitud)'
                
        if altitude_info:
            gps_data['AltitudeInfo'] = altitude_info
    
    return gps_data

def extract_camera_height_data(exif_data):
    """
    Extrae información relacionada con la altura de la cámara desde diferentes fuentes
    """
    height_data = {
        'sources_found': [],
        'height_estimates': {},
        'sensor_data': {},
        'flight_data': {}  # Para drones
    }
    
    # 1. Altitud GPS (altura sobre el nivel del mar)
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        gps_tags = {}
        for key in gps_info.keys():
            name = GPSTAGS.get(key, key)
            gps_tags[name] = gps_info[key]
            
        if 'GPSAltitude' in gps_tags:
            altitude = float(gps_tags['GPSAltitude'])
            altitude_ref = gps_tags.get('GPSAltitudeRef', 0)
            if altitude_ref == 1:
                altitude = -altitude
            height_data['height_estimates']['gps_altitude'] = {
                'value': altitude,
                'unit': 'meters',
                'description': 'Altura sobre el nivel del mar (GPS)',
                'reference': 'Sea Level'
            }
            height_data['sources_found'].append('GPS Altitude')
    
    # 2. Datos de sensores barométricos (algunos smartphones y drones)
    barometric_tags = ['BarometricPressure', 'Pressure', 'RelativeAltitude']
    for tag in barometric_tags:
        if tag in exif_data:
            height_data['sensor_data']['barometric'] = {
                'value': exif_data[tag],
                'tag': tag,
                'description': 'Datos de presión barométrica'
            }
            height_data['sources_found'].append('Barometric Sensor')
    
    # 3. Metadatos específicos de drones (DJI, etc.)
    drone_tags = [
        'DroneAltitude', 'FlightAltitude', 'RelativeAltitude', 
        'AbsoluteAltitude', 'TakeOffAltitude', 'HomePointAltitude'
    ]
    for tag in drone_tags:
        if tag in exif_data:
            height_data['flight_data'][tag.lower()] = {
                'value': exif_data[tag],
                'description': f'Altura del drone: {tag}'
            }
            height_data['sources_found'].append(f'Drone Data ({tag})')
    
    # 4. Metadatos XMP (Adobe, DJI y otros fabricantes)
    xmp_height_indicators = [
        'CameraElevation', 'ShootingHeight', 'AltitudeAboveGround',
        'HeightAboveTerrain', 'FlightHeight'
    ]
    for tag in xmp_height_indicators:
        if tag in exif_data:
            height_data['height_estimates'][tag.lower()] = {
                'value': exif_data[tag],
                'source': 'XMP Metadata',
                'description': f'Altura de cámara: {tag}'
            }
            height_data['sources_found'].append(f'XMP ({tag})')
    
    # 5. Datos de acelerómetro/giroscopio (orientación que puede indicar altura relativa)
    motion_tags = ['Accelerometer', 'Gyroscope', 'Orientation', 'CameraTilt']
    for tag in motion_tags:
        if tag in exif_data:
            height_data['sensor_data']['motion'] = height_data['sensor_data'].get('motion', {})
            height_data['sensor_data']['motion'][tag.lower()] = exif_data[tag]
    
    # 6. Detectar tipo de dispositivo para estimar altura típica
    device_info = {}
    if 'Make' in exif_data and 'Model' in exif_data:
        make = exif_data['Make'].upper()
        model = exif_data['Model'].upper()
        
        # Detectar drones
        drone_brands = ['DJI', 'PARROT', 'AUTEL', 'SKYDIO', 'YUNEEC']
        if any(brand in make for brand in drone_brands):
            device_info['type'] = 'drone'
            device_info['typical_height_range'] = '20-120 meters'
            
        # Detectar cámaras de acción en postes/palos
        elif 'GOPRO' in make or 'INSTA360' in make or 'RICOH THETA' in model:
            device_info['type'] = 'action_camera'
            device_info['typical_height_range'] = '1.5-4 meters'
            
        # Detectar smartphones
        elif any(brand in make for brand in ['APPLE', 'SAMSUNG', 'GOOGLE', 'HUAWEI']):
            device_info['type'] = 'smartphone'
            device_info['typical_height_range'] = '1.2-2 meters'
            
        height_data['device_info'] = device_info
    
    return height_data

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
    Extrae todos los metadatos disponibles de la imagen incluyendo datos de altura
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
        'gps_data': {},
        'camera_height_data': {}  # Nueva sección para datos de altura
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
                
                # Extraer datos de altura de la cámara
                metadata['camera_height_data'] = extract_camera_height_data(metadata['exif_data'])
            
            # Extraer metadatos específicos de 360
            metadata['metadata_360'] = extract_360_metadata(image_path)
            
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None
    
    return metadata

def print_metadata_summary(metadata):
    """
    Imprime un resumen de los metadatos extraídos incluyendo información de altura
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
                print(f"Altitud GPS: {coords['altitude']:.2f} metros sobre el nivel del mar")
            
            # Generar enlace a Google Maps
            lat, lon = coords['latitude'], coords['longitude']
            maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            print(f"Ver en Google Maps: {maps_url}")
    else:
        print("❌ La imagen NO contiene datos de geolocalización")
    
    # Nueva sección: Información de altura de la cámara
    print("\n--- INFORMACIÓN DE ALTURA DE LA CÁMARA ---")
    height_data = metadata.get('camera_height_data', {})
    
    if height_data.get('sources_found'):
        print("✅ Se encontraron datos de altura de la cámara:")
        print(f"Fuentes detectadas: {', '.join(height_data['sources_found'])}")
        
        # Mostrar estimaciones de altura
        if height_data.get('height_estimates'):
            print("\n🔍 Estimaciones de altura:")
            for key, data in height_data['height_estimates'].items():
                print(f"  • {data['description']}: {data['value']} {data.get('unit', 'metros')}")
                if 'reference' in data:
                    print(f"    Referencia: {data['reference']}")
        
        # Mostrar datos de vuelo (drones)
        if height_data.get('flight_data'):
            print("\n🚁 Datos de vuelo de drone:")
            for key, data in height_data['flight_data'].items():
                print(f"  • {data['description']}: {data['value']}")
        
        # Mostrar datos de sensores
        if height_data.get('sensor_data'):
            print("\n📱 Datos de sensores:")
            for sensor_type, data in height_data['sensor_data'].items():
                if sensor_type == 'barometric':
                    print(f"  • Sensor barométrico: {data['description']}")
                elif sensor_type == 'motion':
                    print(f"  • Sensores de movimiento: {list(data.keys())}")
        
        # Información del dispositivo
        if height_data.get('device_info'):
            device = height_data['device_info']
            print(f"\n📷 Tipo de dispositivo detectado: {device.get('type', 'desconocido').upper()}")
            if 'typical_height_range' in device:
                print(f"   Rango típico de altura: {device['typical_height_range']}")
                
    else:
        print("❌ No se encontraron datos específicos de altura de la cámara")
        print("💡 Esto puede deberse a:")
        print("   • La cámara no tiene sensores de altura")
        print("   • Los metadatos no incluyen información de altura")
        print("   • La imagen fue procesada y se perdieron los metadatos")
        
        # Sugerir altura basada en el dispositivo
        if 'Make' in metadata.get('exif_data', {}):
            make = metadata['exif_data']['Make'].upper()
            model = metadata['exif_data'].get('Model', '').upper()
            print(f"\n📱 Dispositivo: {make} {model}")
            
            if 'DJI' in make:
                print("   Estimación: Probablemente tomada entre 20-120 metros (drone)")
            elif 'GOPRO' in make or 'INSTA360' in make:
                print("   Estimación: Probablemente tomada entre 1.5-4 metros (cámara de acción)")
            elif any(brand in make for brand in ['APPLE', 'SAMSUNG', 'GOOGLE']):
                print("   Estimación: Probablemente tomada entre 1.2-2 metros (smartphone)")
    
    # Otros metadatos EXIF relevantes
    print("\n--- OTROS METADATOS RELEVANTES ---")
    relevant_tags = ['DateTime', 'Make', 'Model', 'Software', 'ImageDescription']
    for tag in relevant_tags:
        if tag in metadata['exif_data']:
            print(f"{tag}: {metadata['exif_data'][tag]}")
    
    # Consejos adicionales
    print("\n--- CONSEJOS PARA OBTENER DATOS DE ALTURA ---")
    print("💡 Para obtener datos más precisos de altura:")
    print("   • Usar drones con GPS y barómetro")
    print("   • Activar el GPS en smartphones antes de tomar la foto")
    print("   • Usar aplicaciones especializadas que registren datos de sensores")
    print("   • Verificar que los metadatos no hayan sido eliminados por redes sociales")
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