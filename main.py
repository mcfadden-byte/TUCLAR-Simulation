import warnings

warnings.filterwarnings('ignore')
import os
import queue
import threading
import time

import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import find_optimal
import rule_baseline
import utils_geometry as geo
from agent import AgentHandler
from environment import Environment

import json
import shutil
import sys
import threading
RECORD = False
RECORD_DIR = os.environ.get("LOG_DIR")

HEADLESS = not os.environ.get("DISPLAY")

"""
Simulation: The class that initiates the AgentHandler and the Environment. It is responsible for checking when
            the simulation has succeeded or failed, and handles the visual plot, if there is a display.

Currently, you may see references to a "scenario" parameter. This parameter, as implemented right now,
is almost always meant to be 1. The code and research has overwhelmingly been built around scenario 1, 
the only other scenario that exists right now is 0, which was implemented as a test scenario and not
intended to test any of the things this research is intended to test.
"""
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


    def init_scenario(self):
        self.environment = Environment(self.scenario, self.center_lat, self.center_lon, self.area_size_m)

    def init_handler(self):
        num_agents: int
        match self.scenario:
            case 0: num_agents = 1
            case 1: num_agents = 5
        self.handler = AgentHandler(num_agents, self.scenario, self.environment, 0.8, 1000)
        return True


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
        if self.handler.num_rounds > 500:
            print("TIMEOUT: Simulation has been going on too long. Cutting it off and moving on.")
            return True
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
        self.bounds = (min_lon, max_lon, min_lat, max_lat)

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
        
        # Render obstacles
        self.draw_obstacles()

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


    # Draw each obstacle's 2D footprint once; labeled with its altitude range
    def draw_obstacles(self):
        for obs in self.environment.obstacles:
            ring = []
            for north, east in obs.footprint:
                lat, lon, _ = geo.local_to_latlon(north, east, 0, self.center_lat, self.center_lon)
                ring.append((lon, lat))
            self.ax.add_patch(MplPolygon(ring, closed=True, facecolor="purple", edgecolor="white",
                                         alpha=0.45, zorder=4))
            c = obs.polygon.centroid  # x = north, y = east
            lat, lon, _ = geo.local_to_latlon(c.x, c.y, 0, self.center_lat, self.center_lon)
            self.ax.text(lon, lat, f"{obs.base_alt:.0f}-{obs.top_alt:.0f} m", fontsize=7,
                         color="white", ha="center", va="center", zorder=7)

    def _on_close(self, event):
        self.window_closed = True


    def find_optimal(self):
        optimal = find_optimal.solve(self.environment, self.scenario,
                                     len(self.handler.agents), self.handler.drone_velocity)
        if optimal is None:
            print("No optimal solution found")
        else:
            print(f"Best known solution: {optimal.distance:.0f} m, {optimal.rounds} rounds "
                  f"({optimal.successes}/{optimal.iterations} runs succeeded)")
        return optimal


class Tee:
    """Duplicates a stream (stdout/stderr) into a file. Thread-safe."""

    def __init__(self, stream, file, lock):
        self.stream, self.file, self.lock = stream, file, lock

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.file.write(s)
            self.file.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Recorder:
    """Writes everything an offline replay needs into run_dir:
       scene.json       static scene (bounds, targets, obstacles, config)
       buildings.geojson  building footprints
       rounds.jsonl     one line per round: agent/target state, LLM response times and stats
       frames/round_NNNN.png  plot after each round (round 0 = initial state)
       final.json, log.csv, summary_log.csv (copied)
    """

    def __init__(self, sim, run_dir):
        self.sim = sim
        self.run_dir = run_dir
        self.frames_dir = os.path.join(run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.rounds_file = open(os.path.join(run_dir, "rounds.jsonl"), "w")
        n = len(sim.handler.agents)
        self.seen_stats = [0] * n
        self.seen_times = [0] * n
        self.t0 = time.perf_counter()
        self.save_scene()

    def save_scene(self):
        sim, env, h = self.sim, self.sim.environment, self.sim.handler
        scene = {
            "center_lat": sim.center_lat,
            "center_lon": sim.center_lon,
            "area_size_m": sim.area_size_m,
            "bounds_lon_lat": list(sim.bounds),  # min_lon, max_lon, min_lat, max_lat
            "num_agents": h.num_agents,
            "drone_velocity": h.drone_velocity,
            "comms_quality": h.comms_quality,
            "comms_distance": h.comms_distance,
            "failed_inference_chance": h.failed_inference_chance,
            "targets": [{"north_m": t.north_m, "east_m": t.east_m, "alt": t.alt,
                         "type": t.target_type.value, "required_drones": t.required_drones}
                        for t in env.target_locations],
            "obstacles": [{"footprint_north_east": getattr(o, "footprint", None),
                           "base_alt": getattr(o, "base_alt", None),
                           "top_alt": getattr(o, "top_alt", None)}
                          for o in env.obstacles],
            "config": {k: getattr(agent_mod, k, None) for k in
                       ("PROMPT_STYLE", "PLANNER_MODE", "TARGET_TYPE_VISIBILITY", "LLM_MODEL",
                        "HISTORY_TURNS", "LLM_TIME_LIMIT_SECONDS", "SHOW_OWN_LOCATION",
                        "SHOW_TARGET_COORDS")},
        }
        with open(os.path.join(self.run_dir, "scene.json"), "w") as f:
            json.dump(scene, f, indent=2, default=str)
        try:
            env.buildings_gdf[["geometry"]].to_file(
                os.path.join(self.run_dir, "buildings.geojson"), driver="GeoJSON")
        except Exception as e:
            print(f"WARNING: could not save buildings: {e}")

    def record_round(self):
        h, env = self.sim.handler, self.sim.environment
        n = h.num_rounds  # rounds completed so far
        self.sim.fig.savefig(os.path.join(self.frames_dir, f"round_{n:04d}.png"), dpi=100)

        agents = []
        for a in h.agents:
            i = a.agent_index
            new_stats = a.call_stats[self.seen_stats[i]:]
            new_times = a.response_times[self.seen_times[i]:]
            self.seen_stats[i] = len(a.call_stats)
            self.seen_times[i] = len(a.response_times)
            agents.append({
                "index": i,
                "location": list(a.location),
                "target": a.target_index,
                "inaction": a.inaction,
                "message": a.displayed_message,
                "response_times": new_times,
                "call_stats": new_stats,
            })
        targets = [{"cleared": t.cleared, "visitors": sorted(t.visitors), "visited_by": t.visited_by}
                   for t in env.target_locations]
        self.rounds_file.write(json.dumps({
            "round": n,
            "wall_s": time.perf_counter() - self.t0,
            "agents": agents,
            "targets": targets,
        }, default=str) + "\n")
        self.rounds_file.flush()

    def finish(self, success):
        h = self.sim.handler
        self.rounds_file.close()
        with open(os.path.join(self.run_dir, "final.json"), "w") as f:
            json.dump({"success": success, "num_rounds": h.num_rounds,
                       "sim_time_s": h.get_sim_time(),
                       "avg_response_time_s": h.get_average_response_time()}, f, indent=2)
        for p in (h.log_path, h.summary_log_path):
            if os.path.exists(p):
                shutil.copy(p, self.run_dir)


response_queue = queue.Queue()


def agent_worker(sim, response_queue, render_done=None):
    try:
        while not sim.window_closed:
            if sim.check_success() or sim.check_failure():
                break
            if render_done is not None:
                render_done.clear()
            sim.handler.update_agents()  # only the network-bound work happens here
            response_queue.put("updated")
            if render_done is not None:
                while not render_done.wait(0.5) and not sim.window_closed:
                    pass
    except Exception:
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    session = time.strftime("%Y%m%d_%H%M%S")
    for i in range(20):
        recorder = out_file = None
        real_out, real_err = sys.stdout, sys.stderr
        if RECORD:
            run_dir = os.path.join(RECORD_DIR, session, f"run_{i:03d}")
            os.makedirs(run_dir, exist_ok=True)
            out_file = open(os.path.join(run_dir, "output.txt"), "w")
            lock = threading.Lock()
            sys.stdout = Tee(real_out, out_file, lock)
            sys.stderr = Tee(real_err, out_file, lock)

        try:
            sim = Simulation(scenario=1, center_lat=44.6883889, center_lon=-111.1176389, area_size_m=2500)
            sim.init_scenario()
            sim.init_handler()
            sim.show_map()
            optimal = sim.find_optimal()
            rule = rule_baseline.evaluate(sim.environment, sim.handler.num_agents, sim.handler.drone_velocity, comms_quality=sim.handler.comms_quality, comms_distance=sim.handler.comms_distance, failed_inference_chance=sim.handler.failed_inference_chance, trials=50)

            response_queue = queue.Queue()
            render_done = threading.Event() if RECORD else None
            if RECORD:
                recorder = Recorder(sim, run_dir)
                recorder.record_round()  # round 0: initial state

            worker = threading.Thread(target=agent_worker, args=(sim, response_queue, render_done), daemon=True)
            worker.start()

            # With RECORD the main thread must render every round, even headless.
            if HEADLESS and not RECORD:
                worker.join()
            else:
                while not sim.window_closed and (worker.is_alive() or not response_queue.empty()):
                    try:
                        response_queue.get(timeout=0.1)
                        sim.update_object_positions()
                        if recorder:
                            recorder.record_round()
                            render_done.set()
                    except queue.Empty:
                        pass
                    sim.fig.canvas.flush_events()
                    time.sleep(0.05)
                plt.close(sim.fig)

            success = sim.check_success()
            sim.handler.log_summary(success, optimal, rule)
            print(f"Finished round {i}, Total rounds: {sim.handler.num_rounds}")
            if recorder:
                recorder.finish(success)
        finally:
            if RECORD:
                sys.stdout, sys.stderr = real_out, real_err
                out_file.close()

        if sim.window_closed:
            break
