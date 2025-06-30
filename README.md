# Pipeline de Detección y Geolocalización en Imágenes 360°

Este proyecto procesa imágenes 360° equirectangulares para detectar objetos (árboles), calcular su azimuth y distancia, y obtener coordenadas GPS aproximadas.

## Estructura de Directorios

```
.
├── scripts/                  # Contiene los scripts Python necesarios
│   ├── convert_images.py     # Divide la imagen 360 en cubemap y ejecuta YOLO
│   ├── calculate_azimuths.py # Calcula azimuth de cada detección
│   ├── estimate_distances.py # Estima distancias (m) con UniDepth
│   └── compute_geo_coords.py # Calcula coordenadas GPS por detección
│   └── run_pipeline.py       # Script principal de ejecución automática
│   └── image_gps_data.py     # Extrae datos GPS de las imágenes (SCRIPT DE APOYO, NO SE EJECUTA EN EL run_pipeline.py)
├── modelos/                  # Modelo YOLO (.pt)
│   └── best.pt
├── imagenes/                 # Imágenes 360° de entrada (.jpg, .png)
├── outputs/                  # Resultados (una subcarpeta por imagen)
│   └── output_<imagen>/
│       ├── front.jpg ...     # Caras del cubemap
│       ├── detections.json   # Detecciones YOLO
│       ├── azimuths.json     # Azimuth calculados
│       ├── distances_unidepth.json # Distancias estimadas
│       └── coords.json       # Coordenadas GPS finales
├── logs/                     # Logs de cada ejecución
│   └── logs_<imagen>.txt
└── run_pipeline.py           # Script principal de ejecución automática
```

## Requisitos

- Python 3.7+
- Paquetes: ultralytics, torch, torchvision, opencv-python, Pillow, numpy, einops, huggingface_hub, wandb

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso de `run_pipeline.py`

Ejecuta todo el pipeline en lote o individualmente:

```bash
python run_pipeline.py [OPCIONES]
```

### Argumentos

- `-i, --image`       Procesar solo esta imagen (nombre o ruta). Si no se indica, procesa todas las de `imagenes/`.
- `-c, --cube-size`   Tamaño de cara del cubemap (ej: 512).
- `-m, --model`       Ruta al modelo YOLO (.pt). Default: `modelos/best.pt`.
- `--version`         Versión UniDepth (`v1`, `v2`, `v2old`). Default: `v2`.
- `--backbone`        Backbone UniDepth. Default: `vitl14`.
- `--images-dir`      Directorio de imágenes 360°. Default: `imagenes/`.
- `--scripts-dir`     Directorio de scripts. Default: `scripts/`.
- `--output-root`     Carpeta raíz de resultados. Default: `outputs/`.
- `--logs-dir`        Carpeta raíz de logs. Default: `logs/`.

### Ejemplos

Procesar todas las imágenes:
```bash
python run_pipeline.py
```

Procesar solo `imagen_1.jpg` con cubemap de 512px:
```bash
python run_pipeline.py -i imagen_1.jpg -c 512
```

## Descripción de los Scripts

### `convert_images.py`
1. Carga la imagen equirectangular.
2. Divide en 6 caras de cubemap.
3. Ejecuta detección YOLO en cada cara.
4. Guarda las caras anotadas y `detections.json`.

### `calculate_azimuths.py`
1. Lee `detections.json` y la imagen original.
2. Calcula el azimuth horizontal de cada detección.
3. Genera `azimuths.json`.

### `estimate_distances_unidepth.py`
1. Lee `detections.json` y las caras del cubemap.
2. Usa UniDepth para estimar profundidad (m).
3. Calcula distancia media en cada bounding box.
4. Genera `distances_unidepth.json`.

### `compute_geo_coords.py`
1. Extrae GPS EXIF de la imagen.
2. Lee `azimuths.json` y `distances_unidepth.json`.
3. Calcula coordenadas GPS finales de cada elemento detectado que tenga un bounding box.
4. Genera `coords.json`.
