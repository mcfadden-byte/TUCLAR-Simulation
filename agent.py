from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from environment import Environment
import json
import utils_geometry as geo
from openai import OpenAI
import os
import csv

log_path = "log.csv"
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
        self.agent_index = agent_index


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
        self.location[0] += self.velocity[0]
        self.location[1] += self.velocity[1]
        self.location[2] += self.velocity[2]

    # Scalar distance between an Agent and its target.
    def distance_to_target(self, environment):
        if self.target_index is None:
            return None
        target = environment.target_locations[self.target_index]
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



        self.system_prompt = """
        You must parse the given natural language input for the mission at hand. All inputs will be updates on how the mission is going.
        Your outputs must follow the specified JSON format. Do not write anything EXCEPT this JSON structure in your answer.

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
                    2. You will have a series of teammates. You know where they are, but you cannot communicate with them.
                    3. The mission is a success once all given locations have been visited at least once by you or a teammate.
                    4. Complete the mission as quick as possible. There is no benefit to visiting the same location multiple times.
                    5. Pick ONE location to visit at a time. Your output must follow the JSON format:
                        {
                            "move_to_target": int
                        }

                    Example VALID response: {"move_to_target": 4}
                    Example INVALID response: {"move_to_target": {"index": 7}}
            """
        }

        self.system_prompt += self.scenario_system_prompts[scenario]

        self.num_agents = num_agents
        self.agents = []
        for idx in range(0,num_agents):
            self.agents.append(Agent(system_prompt=self.system_prompt, client=self.client, model=self.model, start_location=[0, 0, 1000], agent_index=idx)) #TODO: Have system that detects ground level?

        self.log_path = log_path
        self.init_log()


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
            self.parse_response(self.agents[idx].generate_response(self.generate_update_message(idx)), idx)

        for agent in self.agents:
            self.update_velocity(agent)
            agent.step()

        self.environment.check_visits(self.agents)
        self.log_round()
        self.num_rounds += 1

    # Interpret the response, search for which target the agent wants to head toward
    def parse_response(self, response: str, agent_index: int):
        print(response)
        agent = self.agents[agent_index]
        try:
            if "{" in response and "}" in response:
                first_open = response.index("{")
                last_close = response.rindex("}")
                response = response[first_open:last_close + 1]
            data = json.loads(response)

            target_index = data.get("move_to_target", agent.target_index)
            if target_index is not None and 0 <= target_index < len(self.environment.target_locations):
                agent.target_index = target_index

        except json.JSONDecodeError:
            print(f"Invalid JSON response: {response}")

    # Set velocity to go towards the target.
    def update_velocity(self, agent):
        if agent.target_index is None:
            agent.velocity = [0, 0, 0]
            return

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
                    f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m, Z={agent.location[2]:.1f}m.\n"
                    f"TARGET LOCATION: X={target_loc[0]:.1f}m, Y={target_loc[1]:.1f}m, Z={target_loc[2]:.1f}m.\n"
                )
            case 1:
                update_message += (
                    f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m, Z={agent.location[2]:.1f}m\n"
                    f"YOUR VELOCITY: X={agent.velocity[0]:.1f}m/s, Y={agent.velocity[1]:.1f}m/s, up={agent.velocity[2]:.1f}m/s\n"
                )

                for idx in range(0, len(self.agents)):
                    if idx == agent_index:
                        continue
                    teammate = self.agents[idx]
                    update_message += (
                        f"TEAMMATE {idx} IS HEADING TO TARGET: {teammate.target_index}, and is {teammate.distance_to_target(self.environment):.0f} meters away.\n"
                    )

                for idx, target in enumerate(self.environment.target_locations):
                    status = "VISITED" if target.visited else "NOT VISITED"
                    update_message += f"TARGET {idx}: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, Z={target.alt:.1f}m, Total Distance: {geo.geometric_mean_3d(agent.location[0], agent.location[1], agent.location[2], target.north_m, target.east_m, target.alt):.0f} meters away. ({status})\n"

        print(update_message)
        print("End AI prompt\n")
        return update_message


    # Return an array of arrays, representing the location of all agents.
    def get_agent_locations(self):
        locations = []
        for agent in self.agents:
            locations.append(agent.location)
        return locations
