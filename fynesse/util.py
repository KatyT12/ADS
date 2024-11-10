def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0):
  distance_lat = distance_km / 111.2
  distance_long = distance_km / (111.32 * math.cos(math.radians(latitude)))

  north = latitude + (distance_lat/2)
  south = latitude - (distance_lat/2)
  west = longitude - (distance_long/2)
  east = longitude + (distance_long/2)
  return (north, south, east, west)