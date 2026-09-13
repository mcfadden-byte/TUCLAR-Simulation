import osmnx as ox
from shapely.geometry import Point
import math
import random

class Environment:

    def __init__(self, scenario: int, center_lat: float, center_lon: float, area_size_m: float = 1000):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.area_size_m = area_size_m

        self.buildings_gdf = None

        self._load_osm_features()

        match scenario:
            case 0: # Scenario 0: One target, random location
                self.target_location = [self.center_lat + random.uniform(-0.002,0.002), self.center_lon + random.uniform(-0.002,0.002), 850]

    # Load map data. Just loads buildings for now.
    def _load_osm_features(self):
        self.buildings_gdf = ox.features_from_point(
            (self.center_lat, self.center_lon),
            tags={"building": True},
            dist=self.area_size_m / math.sqrt(2)
        )

    # Checks if a point is inside a building
    def is_blocked(self, lat, lon):
        point = Point(lon, lat)
        return self.buildings_gdf.intersects(point).any()
