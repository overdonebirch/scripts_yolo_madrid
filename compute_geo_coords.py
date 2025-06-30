#!/usr/bin/env python3
"""
compute_geo_coords.py
Compute GPS coordinates of detected objects given azimuth and distance.
"""
import json
import math
import argparse
from PIL import Image, ExifTags

def destination_point(lat1, lon1, bearing_deg, distance_m, R=6371000):
    # Computes lat2, lon2 given start coords, bearing, and distance
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    theta = math.radians(bearing_deg)
    dr = distance_m / R
    lat2 = math.asin(math.sin(lat1_rad)*math.cos(dr) + math.cos(lat1_rad)*math.sin(dr)*math.cos(theta))
    lon2 = lon1_rad + math.atan2(
        math.sin(theta)*math.sin(dr)*math.cos(lat1_rad),
        math.cos(dr) - math.sin(lat1_rad)*math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


def extract_gps_from_exif(image_path):
    '''Extract (lat, lon) from image EXIF GPS tags'''
    img = Image.open(image_path)
    exif = img._getexif()
    if not exif or 34853 not in exif:
        raise ValueError(f'No GPS EXIF data found in {image_path}')
    gps_info = exif[34853]
    gps_data = {}
    for key, val in gps_info.items():
        name = ExifTags.GPSTAGS.get(key, key)
        gps_data[name] = val
    def to_deg(values):
        # Convert GPS coordinates in EXIF to degrees
        def rat2float(v):
            # Handles PIL IFD Rationals or tuples
            if hasattr(v, 'numerator') and hasattr(v, 'denominator'):
                return v.numerator / v.denominator
            try:
                return v[0] / v[1]
            except:
                return float(v)
        d = rat2float(values[0])
        m = rat2float(values[1])
        s = rat2float(values[2])
        return d + m/60 + s/3600
    lat = to_deg(gps_data['GPSLatitude'])
    if gps_data.get('GPSLatitudeRef','N') != 'N':
        lat = -lat
    lon = to_deg(gps_data['GPSLongitude'])
    if gps_data.get('GPSLongitudeRef','E') != 'E':
        lon = -lon
    return lat, lon

def main():
    parser = argparse.ArgumentParser(description="Compute geo coordinates from azimuths and distances.")
    parser.add_argument("-a", "--azimuths", default="cubemap_output/azimuths.json", help="Azimuths JSON file.")
    parser.add_argument("-d", "--distances", default="cubemap_output/distances_unidepth.json", help="Distances JSON file.")
    parser.add_argument('-i','--image',required=True,help='Equirectangular image with GPS EXIF.')
    parser.add_argument("-o", "--output", default="cubemap_output/coords.json", help="Output JSON file.")
    args = parser.parse_args()
    lat1, lon1 = extract_gps_from_exif(args.image)

    az = json.load(open(args.azimuths))
    dist = json.load(open(args.distances))

    results = {}
    for face, items in az.items():
        face_out = []
        for item in items:
            idx = item.get('bbox_index')
            azimuth = item.get('azimuth_deg')
            # find matching distance
            dval = next((d.get('distance_m') for d in dist.get(face, []) if d.get('bbox_index') == idx), None)
            if dval is None:
                continue
            lat2, lon2 = destination_point(lat1, lon1, azimuth, dval)
            face_out.append({
                'bbox_index': idx,
                'class_id': item.get('class_id'),
                'score': item.get('score'),
                'azimuth_deg': azimuth,
                'distance_m': dval,
                'latitude': lat2,
                'longitude': lon2
            })
        results[face] = face_out

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved coordinates to {args.output}")

if __name__ == '__main__':
    main()
