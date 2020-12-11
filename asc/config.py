class_map = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_station': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9,
}


def get_class_by_index(i):
    for key, value in class_map.items():
        if value == i:
            return key
    return None
