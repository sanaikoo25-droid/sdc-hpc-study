import json

class SDCCompiler:
    def __init__(self, name):
        self.name = name
        self.transitions = []
        self.tile_library = {}

    def add_transition(self, current_state, input_sym, next_state, output_sym):
        self.transitions.append({
            "state_in": current_state,
            "input": input_sym,
            "state_out": next_state,
            "output": output_sym
        })

    def compile_fsm(self, num_positions):
        compiled_tiles = {}
        # Position 0: Anchor
        compiled_tiles[0] = [{
            "tile_id": 0,
            "state_in": "START",
            "state_out": self.transitions[0]["state_in"],
            "type": "anchor"
        }]

        # Positions 1 to num_positions - 1
        for p in range(1, num_positions):
            pos_tiles = []
            for t_idx, trans in enumerate(self.transitions):
                pos_tiles.append({
                    "tile_id": t_idx,
                    "state_in": trans["state_in"],
                    "state_out": trans["state_out"],
                    "input": trans["input"],
                    "output": trans["output"]
                })
            compiled_tiles[p] = pos_tiles

        self.tile_library = {
            "program_name": self.name,
            "scaffold_length": num_positions,
            "tiles_per_position": compiled_tiles
        }
        return self.tile_library

def build_addition_fsm(N=25):
    fsm = SDCCompiler("Full_Binary_Addition")
    rules = [
        (0, (0, 0), 0, 0),
        (0, (0, 1), 0, 1),
        (0, (1, 0), 0, 1),
        (0, (1, 1), 1, 0),
        (1, (0, 0), 0, 1),
        (1, (0, 1), 1, 0),
        (1, (1, 0), 1, 0),
        (1, (1, 1), 1, 1)
    ]
    for c_in, inp, c_out, s in rules:
        fsm.add_transition(c_in, inp, c_out, s)

    return fsm.compile_fsm(num_positions=N)

if __name__ == "__main__":
    compiled_program = build_addition_fsm(N=25)
    with open("compiled_addition_fsm.json", "w") as f:
        json.dump(compiled_program, f, indent=4)
    print("Compiled 25-position ADDITION FSM into compiled_addition_fsm.json")
