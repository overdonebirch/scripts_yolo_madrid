## Insertar metadatos GPS 

exiftool "-GPSLatitude=40.4112325" "-GPSLongitude=-3.6948565" "-GPSLatitudeRef=N" "-GPSLongitudeRef=W" nombre_imagen.jpg

## Obtener azimuths

python calculate_azimuths.py -i nombre_imagen.jpg -d cubemap_output/detections.json -o cubemap_output/azimuths.json

## Obtener distancias

python estimate_distances_unidepth.py -d cubemap_output/detections.json -f cubemap_output -o cubemap_output/distances_unidepth.json --version v2 --backbone vitl14

## Obtener coordenadas

python compute_geo_coords.py -i nombre_imagen.jpg -a cubemap_output/azimuths.json -d cubemap_output/distances_unidepth.json -o cubemap_output/coords.json   