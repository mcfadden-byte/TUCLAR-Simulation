from environment import Environment, TargetType, VISIT_RADIUS_METERS
import json
import utils_geometry as geo
from openai import OpenAI, BadRequestError, APITimeoutError
import os
import csv
import random
import time
from htn_planner import RulePlanner, RuleHTNPlanner, PlanningBudgetExceeded, PlanningFailed
import statistics
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Backend: defaults to a local vLLM server (the cluster). For the hosted NVIDIA API:
#   export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
#   export NVIDIA_API_KEY=nvapi-...
#   export MODEL=nvidia/nemotron-3-ultra-550b-a55b      (optional; this is the default)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.environ.get("NVIDIA_API_KEY", "EMPTY")
LLM_MODEL = os.environ.get("MODEL", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
LLM_TIME_LIMIT_SECONDS = 10
LLM_REMEMBER_REASONING = True
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# Past exchanges each agent remembers (0 = stateless).
HISTORY_TURNS = 3

# CSV logs go here. Set LOG_DIR to write each experiment to its own folder.
LOG_DIR = os.environ.get("LOG_DIR", ".")
log_path = os.path.join(LOG_DIR, "log.csv")
summary_log_path = os.path.join(LOG_DIR, "summary_log.csv")

# --- Experiment configs
PROMPT_STYLE = "short"        # "short": thought/message/action  |  "framework": state/role/task/plan/message/action
SHOW_OWN_LOCATION = False     # tell each agent its own X/Y
SHOW_TARGET_COORDS = False    # tell each agent every pending target's X/Y and distance

# Planner backend (used when PLANNER_MODE == "htn"):
#   "rule"  deterministic role-assignment rules
#   "htn_llm"   determinisitic HTN planning, creating an Area hierarchy. Will delegate to LLM if it takes too long.
#   "llm" llm only, no deterministic planning.
PLANNER_MODE = "llm"
HTN_TIME_BUDGET_SECONDS = 15  # real-world seconds per drone, from its first HTN decision
HTN_BACKEND = "rule"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_SHORT = """
Your output must be in valid JSON format. Do not output anything else.
Your JSON must have "thought", "message", and "action" components.
You can communicate to your teammates through "message", but they may or may not receive it.
In the thought section of the JSON, output the ideal actions for the group as a whole to take, not just yourself.

Example VALID output: {"message": "I will go to target 2, nobody else needs to go there.", "action": "FLY 2"}
Example INVALID output: {"thought": {"reason": "Target 4 is busy", "goal": "Visit all targets"}, "action": {"FLY": 2}}
"""

SYSTEM_PROMPT_FRAMEWORK = """
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
  "message": "<whatever you enter here may or may not reach your teammates. communication is unreliable in this scenario, but try your best to collaborate>",
  "action":  "<copy one action verbatim from your currently-legal actions>"
}
"""

SYSTEM_PROMPT_TOS = """
## Context
You are an agent that is working with other agents. You are working together towards the same goal, but communication is limited.

## How to decide what to do
You will be presented with tasks, some of which may require multiple teammates simultaneously to work on.
You and each of your teammates are a assigned a number, and each task is assigned a number as well.
You are to work on these in the order they are presented. However, assume that your teammates deal with these tasks in order.
For example, if you are Agent 3, and you are presented with 5 available slots for work, you are to take slot 3.
This is because slots 0-2 will have already been claimed by Agents 0-2. Even if they show as empty in the prompt, assume that the change just isn't visible to you yet.

## Reply format
Reply with a single JSON object and nothing else:

{
    "thought": "Enter your reasoning here",
    "message": "Enter a message here. Your teammates may or may not get it; communication is unreliable.",
    "action": "Your choice of action."
}
"""

SYSTEM_PROMPTS = {"short": SYSTEM_PROMPT_SHORT, "framework": SYSTEM_PROMPT_FRAMEWORK, "tos": SYSTEM_PROMPT_TOS}


# Tolerates raw newlines inside JSON strings (a failure the model produces now and then)
_decoder = json.JSONDecoder(strict=False)

def extract_json_object(text):
    """Return the last parseable JSON object containing "action"; ignores braces in reasoning text."""
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _ = _decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    return None

def split_reasoning(reasoning, content):
    if not reasoning and "</think>" in content:
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
    return reasoning.strip(), content.strip()


# Handles LLM calls & message history
class Agent:

    def __init__(self, system_prompt: str, client, model: str, start_location: [float, float, float], agent_index=-1):
        self.history = [{"role": "system", "content": system_prompt}]
        self.recent = []  # last HISTORY_TURNS exchanges, stored compactly
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
        self.displayed_message = "Yet to act, no message for now."
        self.response_times = []
        self.last_stats = []
        self.call_stats = []


        llm_options = dict(max_depth=HTN_MAX_DEPTH, num_candidates=HTN_NUM_CANDIDATES,
                           max_replans=HTN_MAX_REPLANS) if HTN_BACKEND == "llm" else {}

        self.htn = RulePlanner(HTN_TIME_BUDGET_SECONDS)
        self.delegated = False

    def _create(self, messages):
        t0 = time.perf_counter()
        reasoning, content = [], []
        t_first = t_reason_end = None
        finish = usage = stream = None
        timed_out = False
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
                stream=True,
                stream_options={"include_usage": True},
                timeout=LLM_TIME_LIMIT_SECONDS,
            )
            for chunk in stream:
                now = time.perf_counter()
                if now - t0 > LLM_TIME_LIMIT_SECONDS:
                    timed_out = True
                    break
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                d = ch.delta
                finish = ch.finish_reason or finish
                r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if r:
                    t_first = t_first or now
                    reasoning.append(r)
                    t_reason_end = now
                if d.content:
                    t_first = t_first or now
                    content.append(d.content)
                    if t_reason_end is None and "</think>" in "".join(content):
                        t_reason_end = now
        except APITimeoutError:
            timed_out = True
        finally:
            if stream is not None:
                stream.close()
        t_end = time.perf_counter()

        t_first = t_first or t_end
        answer_start = t_reason_end or t_first
        self.last_stats = {
            "ttft": t_first - t0,
            "reasoning_s": (t_reason_end - t_first) if t_reason_end else 0.0,
            "answer_s": t_end - answer_start,
            "total_s": t_end - t0,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "finish_reason": finish,
            "timed_out": timed_out,
        }
        return "".join(reasoning), "".join(content)

    def generate_response(self, message):
        messages = self.history + self.recent + [{"role": "user", "content": message}]
        started = time.perf_counter()
        try:
            reasoning, content = self._create(messages)
        except BadRequestError as e:
            if "context length" in str(e).lower() and self.recent:
                self.recent = []
                reasoning, content = self._create(self.history + [{"role": "user", "content": message}])
            else:
                raise
        self.response_times.append(time.perf_counter() - started)

        if self.last_stats["timed_out"]:
            self.last_stats["invalid_json"] = False  # timeouts are counted separately
            self.call_stats.append(self.last_stats)
            print(f"WARNING: agent {self.agent_index} exceeded {LLM_TIME_LIMIT_SECONDS}s; keeping previous course.")
            return None

        reasoning, content = split_reasoning(reasoning, content)
        parsed = extract_json_object(content)
        self.last_stats["invalid_json"] = parsed is None
        self.call_stats.append(self.last_stats)

        if parsed is not None and HISTORY_TURNS > 0:
            reply = json.dumps({
                "action": parsed.get("action", ""),
                "message": parsed.get("message", ""),
            })
            if LLM_REMEMBER_REASONING and reasoning:
                reply = f"<think>\n{reasoning}\n</think>\n{reply}"
            self.recent += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            self.recent = self.recent[-2 * HISTORY_TURNS:]
        return f"<think>{reasoning}</think>{content}" if reasoning else content

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
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.model = LLM_MODEL

        self.scenario = scenario
        self.environment = environment
        self.num_rounds = 0
        self.comms_quality = 0.5 #Set to 1 for perfect comms
        self.comms_distance = 1000 #Set to -1 for infinite range
        self.drone_velocity = 100
        self.sim_start = None
        self.sim_end = None

        self.system_prompt = SYSTEM_PROMPTS[PROMPT_STYLE]

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
                    2. You will have a series of teammates. You know where they are going, but you cannot communicate with them. All your teammates reason exactly the same as you.
                    3. The mission is a success once all target locations are cleared, and you have used the RECALL action to go back to start.
                    4. There are three types of target location: Sequential, which must be visited by three different agents at any point (you do not need to all visit at the same time). Joint, which must be visited by two different agents at the *same time*. And Exclusive, which must be visited by any agent.
                    5. Complete the mission as quick as possible.
                    6. Pick ONE location to visit at a time. You have 3 available actions:
                            "IDLE": Do nothing.
                            "FLY 0", "FLY 1", "FLY 2"...: Travel to the target with the given index. This is the TARGET index, not the SLOT index.
                            "RECALL": Go back to your starting location.
                    7. It takes multiple rounds for you to travel to a location. You may need to use the FLY action multiple times on the same target before you arrive. Do not assume a target has been visited until it has been cleared from the target list.
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

    def get_sim_time(self):
        """Seconds from the start of the first round to the end of the last one."""
        if self.sim_start is None:
            return 0.0
        return self.sim_end - self.sim_start

    def get_average_response_time(self):
        """Mean LLM response time over every call made by every drone (None if no calls)."""
        times = [t for agent in self.agents for t in agent.response_times]
        return sum(times) / len(times) if times else None

    def log_summary(self, success: bool, optimal=None):
        # total_distance_traveled: Collective distance traveled by all drones
        # revisit_rate: The rate at which drones visited targets they did not need to (rough approximation of poor division of labor)
        # success: Whether or not the simulation succeeded
        # num_rounds: The number of rounds the simulation took to run (Most important for measuring agent problem solving ability)
        # comms_quality: The chance for communications within range to succeed (0-1)
        # sim_time_seconds: The total time the simulation ran, in seconds
        # avg_ttft: Average time-to-first-token for each AI call.
        # std_ttft: Std deviation of times-to-first-token.
        # avg_reasoning_s: Average time spent reasoning for each AI call.
        # std_reasoning_s: Std deviation of reasoning time.
        # avg_answer_s: Average time spent answering for each AI call.
        # std_answer_s: Std deviation of answer time.
        # avg_completion_tokens: Average tokens used for each AI call.
        # std_completion_tokens: Std deviation of completion tokens.
        # optimal_distance: Estimated optimal solution (system is not gaurenteed to find best possible solution)
        # optimal_rounds: The number of rounds the estimated optimal solution took to finish.
        # prompt_style: the prompt style used.
        # planning_style: the planning style used.
        # num_agents: the number of agents in the simulation
        # num_exclusive: The number of exclusive targets.
        # num_joint: The number of joint targets.
        # num_sequential: The number of sequential targets.
        # area_size: Radius of the simulation area, in meters.
        # agent_speed: The maximum distance an agent can move in a round, in meters.
        # num_llm_calls: The number of times an agent was prompted
        # timeout_rate: The frequency that a model timed out.
        # invalid_json_rate: The frequency that a model output in an invalid format
        header = ["total_distance_traveled", "revisit_rate", "success", "num_rounds", "comms_quality",
          "sim_time_seconds", "avg_ttft", "std_ttft", "avg_reasoning_s", "std_reasoning_s",
          "avg_answer_s", "std_answer_s", "avg_completion_tokens", "std_completion_tokens", "optimal_distance",
          "optimal_rounds", "prompt style", "planning style", "num_agents", "num_exclusive", "num_joint", "num_sequential", "area_size", "agent_speed",
          "num_llm_calls", "timeout_rate", "invalid_json_rate"]
        
        avg_response = self.get_average_response_time()
        

        target_types = self.environment.get_target_amounts()
        row = [
            f"{self.get_total_distance_traveled():.2f}",
            f"{self.get_revisit_rate():.4f}",
            success,
            self.num_rounds,
            self.comms_quality,
            f"{self.get_sim_time():.2f}",
            self._avg_stat("ttft"),
            self._std_stat("ttft"),
            self._avg_stat("reasoning_s"),
            self._std_stat("reasoning_s"),
            self._avg_stat("answer_s"),
            self._std_stat("answer_s"),
            self._avg_stat("completion_tokens"),
            self._std_stat("completion_tokens"),
            "" if optimal is None else f"{optimal.distance:.2f}",
            "" if optimal is None else optimal.rounds,
            PROMPT_STYLE,
            PLANNER_MODE,
            self.num_agents,
            target_types[0],
            target_types[1],
            target_types[2],
            self.environment.area_size_m,
            self.drone_velocity,
            self._num_calls(),
            self._rate_stat("timed_out"),
            self._rate_stat("invalid_json")
        ]

        os.makedirs(os.path.dirname(self.summary_log_path) or ".", exist_ok=True)

        # Columns changed since the last run? Keep the old file rather than mixing layouts.
        if os.path.exists(self.summary_log_path):
            with open(self.summary_log_path, newline="") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header != header:
                backup = f"{self.summary_log_path}.{int(time.time())}.bak"
                os.rename(self.summary_log_path, backup)
                print(f"Summary log columns changed; old log moved to {backup}")

        file_exists = os.path.exists(self.summary_log_path)
        with open(self.summary_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def init_log(self):
        header = [f"agent{i}" for i in range(self.num_agents)] + \
                 [f"target{i}" for i in range(len(self.environment.target_locations))]
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
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
        if self.sim_start is None:
            self.sim_start = time.perf_counter()
        # Snapshot who needs a decision before any responses change targets
        to_prompt = [idx for idx in range(self.num_agents) if self.should_prompt(self.agents[idx])]
        print(f"Round {self.num_rounds}: prompting agents {to_prompt}")

        for idx in to_prompt:
            self.decide(idx)

        for idx in range(0, self.num_agents):
            if not self.agents[idx].inaction:
                self.agents[idx].last_target = self.agents[idx].target_index

        for agent in self.agents:
            self.update_velocity(agent)
            agent.step()

        self.environment.check_visits(self.agents)
        self.log_round()
        self.num_rounds += 1
        self.sim_end = time.perf_counter()

    # Choose this drone's action: HTN plan if enabled and within budget, otherwise the LLM
    def decide(self, agent_index: int):
        agent = self.agents[agent_index]
        update_message = self.generate_update_message(agent_index)  # always built and printed

        if PLANNER_MODE == "htn_llm" and self.scenario == 1 and not agent.delegated:
            try:
                kind, target = agent.htn.plan(agent_index, self.num_agents,
                                              self.environment.target_locations,
                                              update_message)
                action = f"FLY {target}" if kind == "FLY" else kind
                print(f"[HTN] drone {agent_index}: {action}")
                reply = {"thought": "HTN plan",
                         "message": f"Executing {action}.",
                         "action": action}
                # Record the exchange so the LLM has context if this drone is delegated later
                if HISTORY_TURNS > 0:
                    agent.recent += [
                        {"role": "user", "content": update_message},
                        {"role": "assistant", "content": json.dumps({"action": action,
                                                                     "message": reply["message"]})},
                    ]
                    agent.recent = agent.recent[-2 * HISTORY_TURNS:]
                self.parse_response(json.dumps(reply), agent_index)
                return
            except (PlanningBudgetExceeded, PlanningFailed) as e:
                agent.delegated = True
                print(f"[HTN] drone {agent_index}: delegating to LLM ({e})")

        self.parse_response(agent.generate_response(update_message), agent_index)


    # Interpret the response, search for which target the agent wants to head toward
    def parse_response(self, response: str, agent_index: int):
        if response is None:
            print("WARNING: Response is None!")
            return
        print(response)
        print()

        agent = self.agents[agent_index]
        data = extract_json_object(response)
        if data is None:
            print(f"Invalid JSON response (tail): {response[-300:]}")
            return

        #IDLE: Set velocity to zero.
        #FLY X: Set velocity similar to previous implementation
        #RECALL: Set inaction to True, set velocity to go back to start.
        action = str(data.get("action", ""))
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

        if random.random() < self.comms_quality:
            agent.displayed_message = data.get("message", "")
        else:
            agent.displayed_message = "Communication Failure."


    # Set velocity to go towards the target.
    def update_velocity(self, agent):
        if agent.target_index is None or agent.target_index == -1:
            agent.velocity = [0, 0, 0]
            return

        if agent.target_index == "Home":
            distance = geo.point_distance(agent.location, [0, 0, 0])
            if distance < 1:
                agent.velocity = [0, 0, 0]
                return
            speed = min(self.drone_velocity, distance)
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
            speed = min(self.drone_velocity, distance)
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
                mates = ", ".join(str(j) for j in range(self.num_agents) if j != agent_index)
                update_message += f"You are **agent {agent_index}**; your teammates are agents {mates}.\n"
                if SHOW_OWN_LOCATION:
                    update_message += f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m\n"

                for idx in range(0, len(self.agents)):
                    if idx == agent_index or self.agents[idx].inaction:
                        continue
                    teammate = self.agents[idx]
                    teammate_target: str
                    teammate_location = self.agents[idx].location
                    if teammate.last_target == -1:
                        teammate_target = "Unknown"
                    elif teammate.last_target is None:
                        teammate_target = "None (idle)"
                    else:
                        mate_target = self.environment.target_locations[teammate.last_target]
                        teammate_target = f"{mate_target.target_type.value} {teammate.last_target}"
                    update_message += f"TEAMMATE {idx} IS HEADING TO TARGET: {teammate_target}\n"
                    
                    if self.comms_distance == -1 or geo.geometric_mean_3d(teammate_location[0], teammate_location[1], teammate_location[2], agent.location[0], agent.location[1], agent.location[2]) <= self.comms_distance:
                        update_message += f"TEAMMATE {idx} SAYS: {self.agents[idx].displayed_message}\n"
                    else:
                        update_message += f"TEAMMATE {idx} SAYS: Out of communication range.\n\n"

                all_targets_visited = all(t.visited for t in self.environment.target_locations)
                if not all_targets_visited:
                    update_message += "\n"

                    # Theory of Scene would encourage a specific format
                    if PROMPT_STYLE == "tos":
                        slot_number = 0  # continues across tasks, starting at 0
                        for idx, target in enumerate(self.environment.target_locations):
                            if target.visited:
                                update_message += f"TASK {idx} COMPLETE\n"
                            else:
                                update_message += f"TASK {idx} ({target.target_type.value}): INCOMPLETE\n"
                                if SHOW_TARGET_COORDS:
                                    distance = geo.point_distance(agent.location, target.get_location())
                                    update_message += (f"    Location: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, "
                                                      f"{distance:.0f} meters away\n")
                                claimants = [a.agent_index for a in self.agents
                                             if a.last_target == idx and not a.inaction]
                                for i in range(target.required_drones):
                                    holder = f"Agent {claimants[i]}" if i < len(claimants) else "Unclaimed"
                                    update_message += f"    {holder}\n"
                                    #update_message += f"    Slot {self.number_to_letters(slot_number)}: {holder}\n"
                                    slot_number += 1

                    # If Theory of Scene is not being used, use a more generic format
                    else:
                        for idx, target in enumerate(self.environment.target_locations):
                            if target.visited:
                                update_message += f"TASK {idx} COMPLETE\n"
                            else:
                                update_message += f"TASK {idx} ({target.target_type.value}): INCOMPLETE\n"
                                if SHOW_TARGET_COORDS:
                                    distance = geo.point_distance(agent.location, target.get_location())
                                    update_message += (f"    Location: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, "
                                                       f"{distance:.0f} meters away\n")

                else:
                    update_message += (
                        "\nAll targets have been visited. The mission is not complete until every drone is back at "
                        "its start location, and IDLE does not move you there. You should now RECALL.\n"
                    )

        print(update_message)
        print("End AI prompt\n")
        return update_message

    def number_to_letters(self, n: int) -> str:
        if n < 0:
            raise ValueError("n must be non-negative")
        result = ""
        n += 1  # shift to 1-indexed for bijective base-26
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(ord('A') + remainder) + result
        return result


    # Return an array of arrays, representing the location of all agents.
    def get_agent_locations(self):
        locations = []
        for agent in self.agents:
            locations.append(agent.location)
        return locations

    # Check if the agents all decided to go home and do nothing
    def check_stalled(self):
        return all(agent.inaction for agent in self.agents)

    def should_prompt(self, agent) -> bool:
        return True
        """Decide whether this agent needs a new decision this round."""
        # Recalled/inactive agents are never prompted
        if agent.inaction or agent.target_index == "Home":
            return False

        # Doing nothing: never assigned (-1) or chose IDLE (None)
        if agent.target_index is None or agent.target_index == -1:
            return True

        target = self.environment.target_locations[agent.target_index]

        # Arrived at its intended location
        if geo.point_distance(agent.location, target.get_location()) < VISIT_RADIUS_METERS:
            return True

        if target.cleared:
            return True

        # Target no longer needs this drone (already cleared, or too many drones assigned)
        return self._is_surplus(agent, target)

    def _is_surplus(self, agent, target) -> bool:
        sequential = target.target_type == TargetType.SEQUENTIAL

        # Sequential counts distinct drones, so a drone that already counted adds nothing by returning
        if sequential and agent.agent_index in target.visitors:
            return True

        rivals = [
            a for a in self.agents
            if a.target_index == agent.target_index
            and not a.inaction
            and not (sequential and a.agent_index in target.visitors)
        ]
        slots = target.slots_remaining()
        if len(rivals) <= slots:
            return False

        # Keep the closest drones on the task; only the surplus (farthest) ones get re-prompted
        target_location = target.get_location()
        rivals.sort(key=lambda a: (geo.point_distance(a.location, target_location), a.agent_index))
        return agent not in rivals[:slots]

    def _avg_stat(self, key):
        vals = [s[key] for a in self.agents for s in a.call_stats if s.get(key) is not None]
        return "" if not vals else f"{sum(vals)/len(vals):.3f}"

    def _std_stat(self, key):
        vals = [s[key] for a in self.agents for s in a.call_stats if s.get(key) is not None]
        return "" if len(vals) < 2 else f"{statistics.stdev(vals):.3f}"

    def _num_calls(self):
        return sum(len(a.call_stats) for a in self.agents)

    def _rate_stat(self, key):
        total = self._num_calls()
        if total == 0:
            return ""
        count = sum(1 for a in self.agents for s in a.call_stats if s.get(key))
        return f"{count / total:.4f}"
