import math

EARTH_RADIUS_M = 6371000  # Earth's mean radius in meters

def degrees_to_meters(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> float:
    """
    Calculate 3D distance between two points on Earth.

    Args:
        lat1, lon1, alt1: Starting point (degrees, meters)
        lat2, lon2, alt2: Ending point (degrees, meters)

    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula for horizontal distance
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    horizontal_distance = EARTH_RADIUS_M * c

    # Vertical distance
    vertical_distance = alt2 - alt1

    # 3D distance
    distance_3d = math.sqrt(horizontal_distance**2 + vertical_distance**2)
    return distance_3d


def meters_to_degrees(lat: float, lon: float, alt: float, bearing: float, distance: float) -> tuple:
    """
    Calculate new point given starting point, bearing, and distance.

    Args:
        lat, lon, alt: Starting point (degrees, meters)
        bearing: Direction in degrees (0-360, where 0/360 is North)
        distance: Displacement in meters (horizontal distance)

    Returns:
        Tuple of (new_lat, new_lon, new_alt)
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)

    # Angular distance in radians
    angular_distance = distance / EARTH_RADIUS_M

    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )

    # Calculate new longitude
    dlon_rad = math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    new_lon_rad = lon_rad + dlon_rad

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)
    new_alt = alt

    return (new_lat, new_lon, new_alt)


def point_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1: Array-like representing coordinates of first point
        point2: Array-like representing coordinates of second point

    Returns:
        Float representing the distance between the points
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")

    sum_of_squares = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return sum_of_squares ** 0.5
