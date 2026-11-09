import heapq

class Action:
    def __init__(self, name, preconditions, add_effects, del_effects):
        self.name = name
        self.preconditions = set(preconditions)
        self.add_effects = set(add_effects)
        self.del_effects = set(del_effects)

    def is_applicable(self, state):
        # Action is applicable if all preconditions are present in the state
        return self.preconditions.issubset(state)

    def apply(self, state):
        # Resulting state = (Current State - Delete Effects) + Add Effects
        return (state - self.del_effects) | self.add_effects

    def __repr__(self):
        return f"Action({self.name})"

class AStarPlanner:
    def __init__(self, actions):
        self.actions = actions

    def heuristic(self, state, goal):
        # Simple heuristic: number of goal variables not yet satisfied
        return len(goal - state)

    def solve(self, initial_state, goal_state):
        # Priority queue stores (priority, cost, current_state, path)
        # priority = cost + heuristic
        start_state = frozenset(initial_state)
        goal_state = frozenset(goal_state)
        
        pq = [(self.heuristic(start_state, goal_state), 0, start_state, [])]
        visited = {start_state: 0}

        while pq:
            priority, cost, state, path = heapq.heappop(pq)

            # Goal check: all goal variables must be true in the state
            if goal_state.issubset(state):
                return path

            for action in self.actions:
                if action.is_applicable(state):
                    next_state = action.apply(state)
                    new_cost = cost + 1
                    
                    if next_state not in visited or new_cost < visited[next_state]:
                        visited[next_state] = new_cost
                        h = self.heuristic(next_state, goal_state)
                        heapq.heappush(pq, (new_cost + h, new_cost, next_state, path + [action.name]))

        return None # No plan found

# Example Usage
if __name__ == "__main__":
    # Scenario: A robot needs to pick up a key and unlock a door.
    # Variables: 'has_key', 'at_door', 'door_open'
    
    actions = [
        Action("pick_up_key", 
               preconditions=[], 
               add_effects=["has_key"], 
               del_effects=[]),
        
        Action("move_to_door", 
               preconditions=[], 
               add_effects=["at_door"], 
               del_effects=[]),
        
        Action("unlock_door", 
               preconditions=["has_key", "at_door"], 
               add_effects=["door_open"], 
               del_effects=["has_key"]), # Consumes the key
    ]

    initial_state = [] # Nothing is true initially
    goal_state = ["door_open"]

    planner = AStarPlanner(actions)
    plan = planner.solve(initial_state, goal_state)

    if plan:
        print(f"Plan found: {' -> '.join(plan)}")
    else:
        print("No plan found.")
