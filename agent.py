from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from environment import Environment
import json
import utils_geometry as geo

# Handles LLM calls & message history
class Agent:

    def __init__(self, system_prompt: str, pipe, generation_args, start_location: [float, float, float]):
        self.history = [{"role": "system", "content": system_prompt}]
        self.pipe = pipe
        self.generation_args = generation_args
        self.location = start_location #lat, lon, alt


    def generate_response(self, message):
        self.history.append({"role": "user", "content": message})
        response = self.pipe(self.history, **self.generation_args)[0]['generated_text']
        self.history.append({"role": "assistant", "content": f"{response.strip()}\n\n"})
        return response


# Handles the LLMs as a group.
class AgentHandler:

    # Initiates the AI, creates the system prompts, initializes variables
    def __init__(self, num_agents: int, scenario: int, environment: Environment):
        self.LLM_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-mini-instruct",
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=False,
                )
        self.LLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", clean_up_tokenization_spaces=False)
        self.pipe = pipeline("text-generation", model=self.LLM_model, tokenizer=self.LLM_tokenizer)
        self.generation_args = {"max_new_tokens": 512, "return_full_text": False, "do_sample": False}

        self.scenario = scenario
        self.environment = environment

        self.system_prompt = """
        You must parse the given natural language input for the mission at hand. All inputs will be updates on how the mission is going.
        Your outputs must follow the specified JSON format. Do not write anything EXCEPT this JSON structure in your answer.

        {
            "action": String ("north", "south", "east", "west", "up", "down", or "none"),
        }

        """

        self.scenario_system_prompts = {

            0: """
                MISSION REQUIREMENTS:
                    1. You will be given a target location and a current location.
                    2. Your goal is to move within 50 meters of the target location, including altitude difference.
                    3. You may only move one direction at a time.
            """
        }

        self.system_prompt += self.scenario_system_prompts[scenario]

        self.num_agents = num_agents
        self.agents = []
        for idx in range(0,num_agents):
            self.agents.append(Agent(self.system_prompt, self.pipe, self.generation_args, [40.4237, -86.9212, 1000]))


    # Generate the next response for every agent
    def update_agents(self):
        for idx in range(0, self.num_agents):
            self.parse_response(self.agents[idx].generate_response(self.generate_update_message(idx)), idx)


    # Take an agent's response, resolve the action it wanted to take
    def parse_response(self, response: str, agent_index: int):
        print(response)
        try:
            if "{" in response and "}" in response:
                first_open = response.index("{")
                last_close = response.rindex("}")
                response = response[first_open:last_close+1]
            data = json.loads(response)
            action = data.get("action", "none")

            match action:
                case "north":
                    self.agents[agent_index].location[0] += 0.00025
                case "south":
                    self.agents[agent_index].location[0] -= 0.00025
                case "east":
                    self.agents[agent_index].location[1] += 0.00025
                case "west":
                    self.agents[agent_index].location[1] -= 0.00025
                case "up":
                    self.agents[agent_index].location[2] += 50
                case "down":
                    self.agents[agent_index].location[2] -= 50
                # "none" does nothing

        except json.JSONDecodeError:
            print(f"Invalid JSON response: {response}")


    # Generate the message updating the agent on the status of the mission
    def generate_update_message(self, agent_index: int):
        update_message = ""
        agent = self.agents[agent_index]

        match self.scenario:
            case 0:
                update_message = geo.get_relative_direction(agent.location, self.environment.target_location)

        print(update_message)
        return update_message


    # Return an array of arrays, representing the location of all agents.
    def get_agent_locations(self):
        locations = []
        for agent in self.agents:
            locations.append(agent.location)
        return locations
