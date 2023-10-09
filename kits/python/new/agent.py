from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import math, random

import sys

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.planned_factories_to_place = 0
        self.metal_per_factory = 150
        self.water_per_factory = 150

    def no_collisions_next_step(self, current_pos, direction, game_state):
        """
        Determines if its safe to move to the next tile without collisions.
        Get positions of all robots from both teams.
        If their current position is on the block of interst, then return False.
        TODO: If their next position is on the block of interst, then return False.
        Else return True.
        """
        target_pos = current_pos + move_deltas[direction]

        my_robots = game_state.units[self.player]
        op_robots = game_state.units[self.opp_player]
        for robot_id, robot in my_robots.items():
            if robot.pos[0] == target_pos[0] and robot.pos[1] == target_pos[1]:
                # next_act = robot.action_queue[0]
                # if next_act 
                return False
        for robot_id, robot in op_robots.items():
            if robot.pos[0] == target_pos[0] and robot.pos[1] == target_pos[1]:
                return False                
        return True

    def move_bot(self, direction, unit, game_state, actions):
        move_cost = unit.move_cost(game_state, direction)
        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
            if self.no_collisions_next_step(unit.pos, direction, game_state):
                actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]
            else:
                other_dirs = [0,1,2,3,4]
                other_dirs.remove(direction)
                random_other_direction = random.choice(other_dirs)
                actions[unit.unit_id] = [unit.move(random_other_direction, repeat=0, n=1)]

    def place_factory(self, obs, game_state, metal_per_factory, water_per_factory):
        potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

        # First modification: lets spawn near ice.
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        min_ice_dist = float(math.inf)
        best_spawn_choice = potential_spawns[np.random.randint(0, len(potential_spawns))]
        for spawn_choice in potential_spawns:
            ice_tile_distances = np.mean((ice_tile_locations - spawn_choice) ** 2, 1)
            # closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
            closest_ice_tile_dist = min(ice_tile_distances)
            if closest_ice_tile_dist <= min_ice_dist:
                best_spawn_choice = spawn_choice
                min_ice_dist = closest_ice_tile_dist
        # spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        spawn_loc = best_spawn_choice
        return dict(spawn=spawn_loc, metal=metal_per_factory, water=water_per_factory)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        original_factories_to_place = 0
        print(f"{step=}",file=sys.stderr) 
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # Initialize
            if self.planned_factories_to_place == 0:
                original_factories_to_place = game_state.teams[self.player].factories_to_place
                self.planned_factories_to_place = max(original_factories_to_place // 2, 1)
                
                # how much water and metal you have in your starting pool to give to new factories
                start_water_left = game_state.teams[self.player].water
                start_metal_left = game_state.teams[self.player].metal
                self.water_per_factory = math.ceil(start_water_left / self.planned_factories_to_place)
                self.metal_per_factory = math.ceil(start_metal_left / self.planned_factories_to_place)
            # factory placement period
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            print(f"{factories_to_place=}, \
                   {self.planned_factories_to_place=}, \
                   {self.water_per_factory=} \
                    {self.metal_per_factory=}", file=sys.stderr)

            if factories_to_place >= self.planned_factories_to_place and my_turn_to_place:
                return self.place_factory(obs, game_state, self.metal_per_factory, self.water_per_factory)
            return dict()
        # else:
        #     factories_to_place = game_state.teams[self.player].factories_to_place
        #     # whether it is your turn to place a factory
        #     my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)

        #     if factories_to_place <= self.planned_factories_to_place and my_turn_to_place:
        #         return self.place_factory(obs, game_state, metal_per_factory, water_per_factory)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            # Try building heavy first. If not enough power, build light.
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                actions[unit_id] = factory.build_light()
            if self.env_cfg.max_episode_length - game_state.real_env_steps < 50:
                if factory.water_cost(game_state) <= factory.cargo.water:
                    actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        
        for unit_id, unit in units.items():
            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 4

                if unit.unit_type=="HEAVY":
                    if unit.cargo.ice < 60:
                        ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                        closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                        if np.all(closest_ice_tile == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        else:
                            direction = direction_to(unit.pos, closest_ice_tile)
                            self.move_bot(direction, unit, game_state, actions)
                    # else if we have enough ice, we go back to the factory and dump it.
                    elif unit.cargo.ice >= 60:
                        direction = direction_to(unit.pos, closest_factory_tile)
                        if adjacent_to_factory:
                            if unit.power >= unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                        else:
                            self.move_bot(direction, unit, game_state, actions)
                                                

                # previous ice mining code
                elif unit.unit_type=="LIGHT": 
                    # Send light ones to the next closest tile and heavies to the closest.
                    # If there isn't enough ice in cargo, go dig.
                    if unit.cargo.ice < 100:
                        ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                        # closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                        sorted_indices = np.argsort(ice_tile_distances)

                        # Take index at position 1 for 2nd closest 
                        next_random_closest_idx = sorted_indices[random.randint(1, int(len(sorted_indices)/2))]
                        if np.all(next_random_closest_idx == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=0, n=1)]
                        else:
                            direction = direction_to(unit.pos, next_random_closest_idx)
                            self.move_bot(direction, unit, game_state, actions)
                    # else if we have enough ice, we go back to the factory and dump it.
                    elif unit.cargo.ice >= 100:
                        direction = direction_to(unit.pos, closest_factory_tile)
                        if adjacent_to_factory:
                            if unit.power >= unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                        else:
                            self.move_bot(direction, unit, game_state, actions)
        return actions
