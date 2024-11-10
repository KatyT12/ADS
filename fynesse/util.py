def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0):
  distance = distance_km / 111.2
  north = latitude + distance/2
  south = latitude - distance/2
  west = longitude - distance/2
  east = longitude + distance/2
  return (north, south, east, west)