import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import contextily as ctx
import pymap3d as pm
from matplotlib.ticker import ScalarFormatter
import utils_geometry as geo
import random
import time
from agent import Agent, AgentHandler

from environment import Environment

class Simulation:

    def __init__(self, scenario: int, center_lat: float, center_lon: float, area_size_m: float = 1000):
        self.scenario = scenario
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.area_size_m = area_size_m

        # Program keeps running when window is closed, use this to stop it
        self.window_closed = False

        self.fig = None
        self.ax = None

        self.environment = None
        self.handler = None

        self.drone_point = None
        self.target_point = None


    # TODO: Expand as the scenarios get more complicated
    def init_scenario(self):
        self.environment = Environment(self.scenario, self.center_lat, self.center_lon, self.area_size_m)

    # TODO: Expand as the scenarios get more complicated
    def init_handler(self):
        self.handler = AgentHandler(1, 0, self.environment)


    # TODO: Expand as the scenarios get more complicated
    def check_success(self):
        match self.scenario:
            case 0:
                target_lat, target_lon, target_alt = self.environment.target_location
                drone_lat, drone_lon, drone_alt = self.handler.get_agent_locations()[0]
                if (geo.degrees_to_meters(target_lat, target_lon, target_alt, drone_lat, drone_lon, drone_alt) < 50):
                    print("SUCCESS!")
                    return True
                print("STILL NOT SUCCESSFUL!")
                return False


    def show_map(self):
        # Define bounds of the map
        half_size = self.area_size_m/2
        min_lat = geo.meters_to_degrees(self.center_lat, self.center_lon, 0, 180, half_size)[0]
        max_lat = geo.meters_to_degrees(self.center_lat, self.center_lon, 0, 0, half_size)[0]
        min_lon = geo.meters_to_degrees(self.center_lat, self.center_lon, 0, 270, half_size)[1]
        max_lon = geo.meters_to_degrees(self.center_lat, self.center_lon, 0, 90, half_size)[1]

        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(min_lon, max_lon)
        self.ax.set_ylim(min_lat, max_lat)
        # Format axes
        self.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.4f}")
        self.ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.4f}")
        # Set title
        self.ax.set_title(f"Simulation area: {self.center_lat:.5f}, {self.center_lon:.5f}")

        # Display satellite image of the location at the given coords
        ctx.add_basemap(self.ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery)

        # Display buildings loaded from OSM
        self.environment.buildings_gdf.plot(ax=self.ax, facecolor="orange", edgecolor="black", alpha=0.5, zorder=3)

        self.update_object_positions()

        # When the window closes, run _on_close
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        plt.ion()
        plt.show()
        plt.pause(0.1)


    # Update positions of all objects represented as points (Drones, target, etc.)
    def update_object_positions(self):
        if self.drone_point is not None:
            self.drone_point.remove()
        if self.target_point is not None:
            self.target_point.remove()

        drone_lat, drone_lon, drone_alt = self.handler.get_agent_locations()[0]
        self.drone_point = self.ax.scatter(drone_lon, drone_lat, color="c", zorder=6, s=100)

        target_lat, target_lon, target_alt = self.environment.target_location
        self.target_point = self.ax.scatter(target_lon, target_lat, color="r", zorder=5, s=100)

        plt.pause(0.01)


    def _on_close(self, event):
        self.window_closed = True



if __name__ == "__main__":
    sim = Simulation(scenario=0, center_lat=40.4237, center_lon=-86.9212, area_size_m=1000)  # Purdue University
    sim.init_scenario()
    sim.init_handler()
    sim.show_map()

    while not sim.check_success() and not sim.window_closed:
        sim.handler.update_agents()
        sim.update_object_positions()
