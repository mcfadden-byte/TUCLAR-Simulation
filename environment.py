import osmnx as ox
from shapely.geometry import Point
import math
import random
import utils_geometry as geo
from enum import Enum

class TargetType(Enum):
    EXCLUSIVE = "Exclusive"
    JOINT = "Joint"
    SEQUENTIAL = "Sequential"

# Number of drones required to clear each type
REQUIRED_DRONES = {
    TargetType.EXCLUSIVE: 1,
    TargetType.JOINT: 2,
    TargetType.SEQUENTIAL: 3,
}


VISIT_RADIUS_METERS = 100 #TODO: Adjust as appropriate

class TargetLocation:

    def __init__(self, north_m: float, east_m: float, alt: float = 1000, target_type: TargetType = TargetType.EXCLUSIVE):
        self.north_m = north_m # Distance North of the center
        self.east_m = east_m # Distance East of center
        self.alt = alt # Altitude, in meters ASL
        self.target_type = target_type
        self.required_drones = REQUIRED_DRONES[target_type]
        self.cleared: bool = False
        self.scatter_pointer = None # A pointer to this target's associated scatter point on the map. Assigned in main.
        self.visited_by: int = -1   # last drone to visit
        self.visitors = set()       # every distinct drone that has ever visited

    # Alias so existing code that reads target.visited keeps working
    @property
    def visited(self):
        return self.cleared

    def get_location(self):
        return self.north_m, self.east_m, self.alt

    def record_visits(self, agents_in_range: list):
        """Called once per round with the indices of all drones currently within range."""
        if not agents_in_range:
            return

        self.visitors.update(agents_in_range)
        self.visited_by = agents_in_range[-1]

        if self.cleared:
            return

        match self.target_type:
            case TargetType.EXCLUSIVE:
                self.cleared = len(agents_in_range) >= 1
            case TargetType.JOINT:
                self.cleared = len(agents_in_range) >= 2      # simultaneous, this round
            case TargetType.SEQUENTIAL:
                self.cleared = len(self.visitors) >= 3        # cumulative, distinct drones

    def slots_remaining(self):
        """How many more drones this target can usefully absorb."""
        if self.cleared:
            return 0
        if self.target_type == TargetType.SEQUENTIAL:
            return self.required_drones - len(self.visitors)  # distinct drones still needed
        return self.required_drones

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
                self._spawn_targets({
                    TargetType.EXCLUSIVE: random.randint(2, 5),
                    TargetType.JOINT: random.randint(1, 4),
                    TargetType.SEQUENTIAL: random.randint(2, 4),
                })

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
        for target in self.target_locations:
            in_range = [
                agent.agent_index for agent in agents
                if geo.point_distance(agent.location, target.get_location()) < VISIT_RADIUS_METERS
            ]
            target.record_visits(in_range)

    def all_targets_visited(self):
        return all(target.cleared for target in self.target_locations)


    def check_visit(self, lat: float, lon: float, alt: float):
        for location in self.target_locations:
            lat1, lon1, alt1 = location.get_location()
            if degrees_to_meters(lat1, lon1, alt1, lat, lon, alt) < VISIT_RADIUS_METERS:
                return True
        return False

    def get_target_locations(self):
        result = []
        for target in self.target_locations:
            result.append(target.get_location())
        return result

    def _spawn_targets(self, counts: dict):
        half = self.area_size_m / 2
        for target_type, count in counts.items():
            for _ in range(count):
                self.target_locations.append(
                    TargetLocation(random.uniform(-half, half), random.uniform(-half, half), target_type=target_type)
                )
        random.shuffle(self.target_locations)

    def get_target_amounts(self):
        result = [0, 0, 0]
        for target in self.target_locations:
            match target.target_type:
                case TargetType.EXCLUSIVE: result[0] += 1
                case TargetType.JOINT: result[1] += 1
                case TargetType.SEQUENTIAL: result[2] += 2

        return result

