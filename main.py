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
import threading
import queue


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
        num_agents: int
        match self.scenario:
            case 0: num_agents = 1
            case 1: num_agents = 5
        self.handler = AgentHandler(num_agents, self.scenario, self.environment)


    # TODO: Expand as the scenarios get more complicated
    def check_success(self):
        match self.scenario:
            case 0:
                target_lat, target_lon, target_alt = self.environment.target_locations[0].get_location()
                drone_lat, drone_lon, drone_alt = self.handler.get_agent_locations()[0]
                if (geo.degrees_to_meters(target_lat, target_lon, target_alt, drone_lat, drone_lon, drone_alt) < 50):
                    print("SUCCESS!")
                    return True
                print("STILL NOT SUCCESSFUL!")
                return False
            case 1:
                if self.check_all_agents_home() and self.check_all_targets_visited():
                    print(f"SUCCESS! Completed in {self.handler.num_rounds} rounds.\n")
                    return True
                print(f"STILL NOT SUCCESSFUL! (round {self.handler.num_rounds})\n")
                return False

    def check_all_agents_home(self):
        return all(
                    geo.point_distance(agent.location, [0, 0, 0]) < 1
                    for agent in self.handler.agents
                )

    def check_all_targets_visited(self):
        return self.environment.all_targets_visited()

    # If all agents decided to go home and do nothing before the mission was complete, the mission is a failure.
    def check_failure(self):
        match self.scenario:
            case 1:
                if self.handler.check_stalled() and not self.check_all_targets_visited():
                    print(f"MISSION FAILED: all agents inactive, mission incomplete. (round {self.handler.num_rounds})\n")
                    return True
                return False
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
        plt.draw()
        plt.pause(0.1)


    # Update positions of all objects represented as points (Drones, target, etc.)
    def update_object_positions(self):
        for location in self.environment.target_locations:
            if location.scatter_pointer is not None:
                location.scatter_pointer.remove()
        for agent in self.handler.agents:
            if agent.scatter_pointer is not None:
                agent.scatter_pointer.remove()


        for agent in self.handler.agents:
            lat, lon, _ = geo.local_to_latlon(agent.location[0], agent.location[1], agent.location[2], self.center_lat, self.center_lon)
            agent.scatter_pointer = self.ax.scatter(lon, lat, color="c", zorder=6, s=100)


        for target in self.environment.target_locations:
            color = "gray" if target.visited else "r"
            lat, lon, _ = geo.local_to_latlon(target.north_m, target.east_m, target.alt, self.center_lat, self.center_lon)
            target.scatter_pointer = self.ax.scatter(lon, lat, color=color, zorder=5, s=100)


        plt.pause(0.01)


    def _on_close(self, event):
        self.window_closed = True



response_queue = queue.Queue()


def agent_worker():
    while not sim.window_closed:
        if sim.check_success() or sim.check_failure():
            break
        sim.handler.update_agents()  # only the network-bound work happens here
        response_queue.put("updated")


if __name__ == "__main__":
    # Metrics we'll use to measure the quality of success
    num_rounds: int
    num_visits: int


    sim = Simulation(scenario=1, center_lat=44.6883889, center_lon=-111.1176389, area_size_m=2500)
    sim.init_scenario()
    sim.init_handler()
    sim.show_map()

    worker = threading.Thread(target=agent_worker, daemon=True)
    worker.start()

    # Simulation loop: simulates in rounds, where the drones can move a certain distance each round.
    # Check if success condition has been met (break if true)
    # If a drone is not in the middle of a task, ask the associated agent for the next task.
    # Redraw the map
    while not sim.window_closed:
        try:
            response_queue.get(timeout=0.1)
            sim.update_object_positions()
        except queue.Empty:
            pass
        sim.fig.canvas.flush_events()
        time.sleep(0.05)

    success = sim.check_success()
    sim.handler.log_summary(success)
    print(f"Total rounds: {sim.handler.num_rounds}")
