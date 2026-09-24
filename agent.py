from environment import Environment
import json
import utils_geometry as geo
from openai import OpenAI
import os
import csv

log_path = "log.csv"
summary_log_path = "summary_log.csv"
VELOCITY_METERS_SECOND = 100


# Handles LLM calls & message history
class Agent:

    def __init__(self, system_prompt: str, client, model: str, start_location: [float, float, float], agent_index=-1):
        self.history = [{"role": "system", "content": system_prompt}]
        self.client = client
        self.model = model
        self.location = start_location # Distance North from center, distance East from center, altitude
        self.velocity = [0,0,0] # North velocity (m/s), East velocity (m/s), Upward velocity (m/s)
        self.scatter_pointer = None # A pointer to this agent's associated drone on the map. Assigned in main.
        self.target_index = -1
        self.last_target = -1
        self.agent_index = agent_index
        self.inaction = False
        self.distance_traveled = 0.0


    def generate_response(self, message):
        self.history.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": f"{content.strip()}\n\n"})
        return content

    # Advance position according to velocity
    def step(self):
        step_distance = geo.point_distance([0, 0, 0], self.velocity)  # magnitude of velocity this round
        self.distance_traveled += step_distance
        self.location[0] += self.velocity[0]
        self.location[1] += self.velocity[1]
        self.location[2] += self.velocity[2]

    # Scalar distance between an Agent and its target.
    def distance_to_target(self, environment):
        if self.target_index is None:
            return None
        target = environment.target_locations[self.target_index]
        return geo.point_distance(self.location, target.get_location())

    # Scalar distance between an Agent and its previous target.
    def distance_to_last_target(self, environment):
        if self.last_target is None:
            return None
        target = environment.target_locations[self.last_target]
        return geo.point_distance(self.location, target.get_location())


# Handles the LLMs as a group.
class AgentHandler:

    # Initiates the AI, creates the system prompts, initializes variables
    def __init__(self, num_agents: int, scenario: int, environment: Environment):
        #self.LLM_model = AutoModelForCausalLM.from_pretrained(
        #        "microsoft/Phi-4-mini-instruct",
        #        device_map="auto",
        #        torch_dtype="auto",
        #        trust_remote_code=False,
        #        )
        #self.LLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", clean_up_tokenization_spaces=False)
        #self.pipe = pipeline("text-generation", model=self.LLM_model, tokenizer=self.LLM_tokenizer)
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        #"nvidia/nemotron-3.5-lightning-30b-a3b"
        #mistralai/mistral-nemotron
        self.model = "nvidia/nemotron-3.5-lightning-30b-a3b"

        self.generation_args = {"max_tokens": 512, "return_full_text": False, "temperature": False}

        self.scenario = scenario
        self.environment = environment
        self.num_rounds = 0


#        self.system_prompt = """
#            Your output must be in valid JSON format. Do not output anything else.
#            Your JSON must have "thought" and "action" components.

#            Example VALID output: {"thought": "Teammate 3 is already going to target 4, so I will go to target 2", "action": "FLY 2"}
#            Example INVALID output: {"thought": {"reason": "Target 4 is busy", "goal": "Visit all targets"}, "action": {"FLY": 2}}
#        """

        self.system_prompt = """
## Coordinating from the shared observation
Decide from the observation you and the other agents all see: reading the same scene and reasoning alike, a rule anchored to that scene leads you all to the same division of labor.

**Role.** Read the ownership: is it overlapping -- do you and the other agents all work the same targets and resources -- or is it divided, each of you already holding its own part?
  - **Overlapping** (acting alike you would contend for one target or all defer): coordinate -- settle a division of labor over the shared targets from the scene.
  - **Divided** (your role owns some stages/stations and the other agents own the rest): hold your part and execute. Take the single most useful action within your own part; do not step onto a station or task that belongs to another agent's part, even if it looks like the most useful step right now -- reading the same scene, they are already taking it, so you would only collide. If your own part has no ready step (it waits on their output), do its enabling step or start the next independent unit; wait only when nothing of yours is productive.

**Task.** Settle the division over the shared targets: read how each pending target couples you and apply its rule over the canonical order:
  1. **Joint** (succeeds only if you all act on it together): converge -- all take the same one: the first such target in that order. (Acting alone wastes it.)
  2. **Exclusive** (single-occupancy -- multiple acting on it wastes it): divide by item -- take your share by role over that order; the other agents take the complement, so you never collide.
  3. **Sequential** (ordered hand-off chain): divide by stage -- work the stage your role owns, by the chain order.

## Reply format
Reply with a single JSON object and nothing else:

{
  "state":   "<from the observation only: what you hold + which steps are doable now (inputs present); world state, never the other agents' intent>",
  "role":    "<is ownership overlapping (you and the other agents all work the same targets/resources -> coordinate) or divided (your role owns some stages/stations, the others own the rest -> execute your own), and why>",
  "task":    "<for each pending shared target, how it couples you -- Joint / Exclusive / Sequential>",
  "plan":    "<overlapping: apply the matching rule per target over the canonical order -- Joint: converge on the first such target; Exclusive: your role's share; Sequential: your stage; divided: the most useful step within your own part (enabling step or next unit if none ready), not a step that belongs to another agent's part>",
  "action":  "<copy one action verbatim from your currently-legal actions>"
}\n

        """

        self.scenario_system_prompts = {

            0: """
                MISSION REQUIREMENTS:
                    1. You will be given a target location and a current location.
                    2. Your goal is to move within 50 meters of the target location, including altitude difference.
                    3. You may only move one direction at a time.
            """,
            1: """
                MISSION REQUIREMENTS:
                    1. There is a series of locations that you will be given.
                    2. You will have a series of teammates. You know where they are going, but you cannot communicate with them.
                    3. The mission is a success once all given locations have been visited at least once by you or a teammate, AND all drones have gone back to their start location.
                    4. Complete the mission as quick as possible. There is no benefit to visiting the same location multiple times.
                    5. Pick ONE location to visit at a time. You have 3 available actions:
                            "IDLE": Do nothing.
                            "FLY 0", "FLY 1", "FLY 2"...: Travel to the target with the given index.
                            "RECALL": Go back to your starting location.
                    6. It takes multiple rounds for you to travel to a location. You may need to use the FLY action multiple times on the same target before you arrive.
            """
        }

        self.system_prompt += self.scenario_system_prompts[scenario]

        self.num_agents = num_agents
        self.agents = []
        for idx in range(0,num_agents):
            self.agents.append(Agent(system_prompt=self.system_prompt, client=self.client, model=self.model, start_location=[0, 0, 1000], agent_index=idx)) #TODO: Have system that detects ground level?

        self.log_path = log_path
        self.summary_log_path = summary_log_path
        self.init_log()


    def get_total_distance_traveled(self):
        return sum(agent.distance_traveled for agent in self.agents)

    def get_revisit_rate(self):
        targets = self.environment.target_locations
        if not targets:
            return 0.0
        revisited = sum(1 for t in targets if len(t.visitors) > 1)
        return revisited / len(targets)

    def log_summary(self, success: bool):
        header = ["total_distance_traveled", "revisit_rate", "success", "num_rounds"]
        row = [
            f"{self.get_total_distance_traveled():.2f}",
            f"{self.get_revisit_rate():.4f}",
            success,
            self.num_rounds,
        ]
        file_exists = os.path.exists(self.summary_log_path)
        with open(self.summary_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def init_log(self):
        header = [f"agent{i}" for i in range(self.num_agents)] + \
                 [f"target{i}" for i in range(len(self.environment.target_locations))]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_round(self):
        agent_targets = [
            agent.target_index if agent.target_index is not None else -1
            for agent in self.agents
        ]

        target_visitors = []
        for target in self.environment.target_locations:
            visitor = getattr(target, "visited_by", -1)
            target_visitors.append(visitor)

        row = agent_targets + target_visitors
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # Update all agents on the situation, ask for a new course of action (or continue the current one)
    def update_agents(self):
        for idx in range(0, self.num_agents):
            if not self.agents[idx].inaction:
                self.parse_response(self.agents[idx].generate_response(self.generate_update_message(idx)), idx)

        for idx in range(0, self.num_agents):
            if not self.agents[idx].inaction:
                self.agents[idx].last_target = self.agents[idx].target_index

        for agent in self.agents:
            self.update_velocity(agent)
            agent.step()

        self.environment.check_visits(self.agents)
        self.log_round()
        self.num_rounds += 1

    # Interpret the response, search for which target the agent wants to head toward
    def parse_response(self, response: str, agent_index: int):
        print(response)
        print()

        agent = self.agents[agent_index]
        try:
            if "{" in response and "}" in response:
                first_open = response.index("{")
                last_close = response.rindex("}")
                response = response[first_open:last_close + 1]
            data = json.loads(response)

            #IDLE: Set velocity to zero.
            #FLY X: Set velocity similar to previous implementation
            #RECALL: Set inaction to True, set velocity to go back to start.
            action = data.get("action", "")
            if action == "IDLE":
                agent.target_index = None
            elif action == "RECALL":
                agent.target_index = "Home"
                agent.inaction = True
            elif action.startswith("FLY "):
                try:
                    index = int(action.removeprefix("FLY "))
                    if 0 <= index < len(self.environment.target_locations):
                        agent.target_index = index
                    else:
                        print(f"Agent {agent_index} gave out-of-range FLY index: {index}")
                except ValueError:
                    print(f"Agent {agent_index} gave malformed FLY action: {action!r}")
            else:
                print(f"Agent {agent_index} gave unrecognized action: {action!r}")

        except json.JSONDecodeError:
            print(f"Invalid JSON response: {response}")

    # Set velocity to go towards the target.
    def update_velocity(self, agent):
        if agent.target_index is None:
            agent.velocity = [0, 0, 0]
            return

        if agent.target_index == "Home":
            distance = geo.point_distance(agent.location, [0, 0, 0])
            if distance < 1:
                agent.velocity = [0, 0, 0]
                return
            speed = min(VELOCITY_METERS_SECOND, distance)
            scale = speed / distance
            agent.velocity = [-d * scale for d in agent.location]
            return

        if not isinstance(agent.target_index, int):
            print(f"WARNING: Non-integer target index: {agent.target_index}")
        else:
            target = self.environment.target_locations[agent.target_index]
            target_location = target.get_location()

            distance = geo.point_distance(agent.location, target_location)
            if distance < 1e-6:
                agent.velocity = [0, 0, 0]
                return

            delta = [target_location[i] - agent.location[i] for i in range(3)]
            speed = min(VELOCITY_METERS_SECOND, distance)
            scale = speed / distance
            agent.velocity = [d * scale for d in delta]



    # Generate the message updating the agent on the status of the mission
    def generate_update_message(self, agent_index: int):
        agent = self.agents[agent_index]
        update_message = ""

        print(f"Message sent to Agent {agent_index}:")

        match self.scenario:
            case 0:
                target_loc = self.environment.target_locations[0].get_location()
                update_message = (
                    f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m.\n"
                    f"TARGET LOCATION: X={target_loc[0]:.1f}m, Y={target_loc[1]:.1f}m.\n"
                )
            case 1:
                update_message += (
                    f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m\n"
                    #f"YOUR VELOCITY: X={agent.velocity[0]:.1f}m/s, Y={agent.velocity[1]:.1f}m/s\n"
                )

                for idx in range(0, len(self.agents)):
                    if idx == agent_index or self.agents[idx].inaction:
                        continue
                    teammate = self.agents[idx]
                    teammate_target: str
                    if teammate.last_target == -1:
                        teammate_target = "Unknown"
                    else:
                        teammate_target = f"{teammate.last_target}"#, and is {teammate.distance_to_last_target(self.environment):.0f} meters away."
                    update_message += (
                        f"TEAMMATE {idx} IS HEADING TO TARGET: {teammate_target}\n"
                    )
                all_targets_visited: True
                for idx, target in enumerate(self.environment.target_locations):
                    if target.visited == False:
                        all_targets_visited = False
                        break
                if not all_targets_visited:
                    update_message += "\nThe following targets have yet to be visited:\n"
                    for idx, target in enumerate(self.environment.target_locations):
                        if target.visited:
                            continue
                        #status = "VISITED" if target.visited else "NOT VISITED"
                        update_message += f"TARGET {idx}: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, Total Distance: {geo.geometric_mean_3d(agent.location[0], agent.location[1], agent.location[2], target.north_m, target.east_m, target.alt):.0f} meters away.\n"
                else:
                    update_message += "\nAll targets have been visited.\n"

        print(update_message)
        print("End AI prompt\n")
        return update_message


    # Return an array of arrays, representing the location of all agents.
    def get_agent_locations(self):
        locations = []
        for agent in self.agents:
            locations.append(agent.location)
        return locations

    # Check if the agents all decided to go home and do nothing
    def check_stalled(self):
        return all(agent.inaction for agent in self.agents)
