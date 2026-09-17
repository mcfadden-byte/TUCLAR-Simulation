import osmnx as ox
from shapely.geometry import Point
import math
import random
import utils_geometry as geo

VISIT_RADIUS_METERS = 100 #TODO: Adjust as appropriate

class TargetLocation:

    def __init__(self, north_m: float, east_m: float, alt: float=1000):
        self.north_m = north_m # Distance North of the center
        self.east_m = east_m # Distance East of center
        self.alt = alt # Altitude, in meters ASL
        self.visited: bool = False
        self.scatter_pointer = None # A pointer to this agent's associated scatter point on the map. Assigned in main.
        self.visited_by: int = -1

    #def get_location_degrees(self):
    #    return self.lat, self.lon, self.alt

    def get_location(self):
        return self.north_m, self.east_m, self.alt

class Environment:



    def __init__(self, scenario: int, center_lat: float, center_lon: float, area_size_m: float = 1000):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.area_size_m = area_size_m

        self.buildings_gdf = None

        self._load_osm_features()

        self.target_locations = []

        match scenario:
            case 0: # Scenario 0: One target, random location
                self.target_locations.append(TargetLocation(random.uniform(-area_size_m, area_size_m), random.uniform(-area_size_m, area_size_m)))
            case 1:
                #TODO Replace with forest spots?
                for i in range(0, 5):
                    self.target_locations.append(TargetLocation(random.uniform(-area_size_m/2, area_size_m/2), random.uniform(-area_size_m/2, area_size_m/2)))

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

    def check_visits(self, agents):
        for agent in agents:
            for location in self.target_locations:
                [lat1, lon1, alt1] = location.get_location()
                if geo.geometric_mean_3d(lat1, lon1, alt1, agent.location[0], agent.location[1], agent.location[2]) < VISIT_RADIUS_METERS:
                    location.visited = True
                    location.visited_by = agent.agent_index
                    break


    def check_visit(self, lat: float, lon: float, alt: float):
        for location in self.target_locations:
            lat1, lon1, alt1 = location.get_location()
            if degrees_to_meters(lat1, lon1, alt1, lat, lon, alt) < VISIT_RADIUS_METERS:
                return True
        return False


    def all_targets_visited(self):
        result: bool = True
        for target in self.target_locations:
            if target.visited is False:
                result = False
                break

        return result
