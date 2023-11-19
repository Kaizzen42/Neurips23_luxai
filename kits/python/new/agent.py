from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import math, random
from collections import defaultdict
import sys
import re
import networkx as nx

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
opposite_directions = {1:3,3:1,2:4,4:2,0:0}


def pp(strp,sep='\n'):
    return print(strp,sep=sep,file=sys.stderr)

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.planned_factories_to_place = 0
        self.metal_per_factory = 150
        self.water_per_factory = 150
        self.ice_mining_target_map = dict() # dict to map robot to its mining target
        self.ice_mining_target_reverse_map = dict() # dict to map mining target tile to robot id
        self.ore_mining_target_map = dict() # dict to map robot to its mining target
        self.ore_mining_target_reverse_map = dict() # dict to map mining target tile to robot id
        self.total_original_water = 0
        self.bot_role = dict()  # Every bot would be classified as miner/cleaner/redbot (enemy focussed)
        self.attack_target_map = dict() # maps redbot to target tile
        self.attack_target_reverse_map = dict() 
        self.ATTACK_MODE = False # Turned on when ready to attack opponent lichens.
        self.rsc_pths = dict() # key = (factory_tile, resource_tile), value = [steps]
        """
        1. Generate k optimal paths (distance wise) from src to dest.
        2. Weigh them by rubble on them, and find the most optimal path.
        3. Sent dig bots to clear the path.
        4. Build elastic logic (reeled back when energy < return_energy)
        """

    def no_collisions_next_step(self, current_pos, unit, direction, game_state):
        """
        Determines if its safe to move to the next tile without collisions.
        Get positions of all robots from both teams.
        If their current position is on the block of interst, then return False.
        TODO: If their next position is on the block of interst, then return False.
        Else return True.
        """
        target_pos = current_pos + move_deltas[direction]
        unit_type = unit.unit_type
        my_robots = game_state.units[self.player]
        op_robots = game_state.units[self.opp_player]
        for robot_id, robot in my_robots.items():
            # if robot.pos[0] == target_pos[0] and robot.pos[1] == target_pos[1]:
            #     return False
            next_pos = self.get_next_queue_position(robot, robot.action_queue)
            if next_pos[0] == target_pos[0] and next_pos[1] == target_pos[1]:
                return False
        for robot_id, robot in op_robots.items():
            next_pos = self.get_next_queue_position(robot, robot.action_queue)
            if unit_type=="HEAVY":
                if robot.unit_type=="LIGHT":
                    if next_pos[0] == target_pos[0] and next_pos[1] == target_pos[1]:
                        return True
                else:
                    if robot.power < unit.power:
                        if next_pos[0] == target_pos[0] and next_pos[1] == target_pos[1]:
                            return True


            else:    
                if next_pos[0] == target_pos[0] and next_pos[1] == target_pos[1]:
                    return False
        return True

    def move_bot(self, direction, unit, game_state, actions):
        """ Moves the bot one step in the given direction if there are no collisions.
            If there are collisions it picks a direction orthogonal to the collision direction.
            If it collides in both orthogonal directions then it stays there and recharges.
        """
        move_cost = unit.move_cost(game_state, direction)
        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
            if self.no_collisions_next_step(unit.pos,unit, direction, game_state):
                actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]
            else:
                # Lets move in one of the orthogonal directions.
                other_dirs = [0,1,2,3,4]
                other_dirs.remove(direction)
                opposite_dir = opposite_directions[direction]
                if opposite_dir != 0:
                    other_dirs.remove(opposite_dir)

                random_other_direction = random.choice(other_dirs)
                if self.no_collisions_next_step(unit.pos, unit, random_other_direction, game_state):
                    actions[unit.unit_id] = [unit.move(random_other_direction, repeat=0, n=1)]
                else: # try moving opposite to the orthogonal direction
                    if self.no_collisions_next_step(unit.pos, unit, opposite_directions[random_other_direction], game_state):
                        actions[unit.unit_id] = [unit.move(opposite_directions[random_other_direction], repeat=0, n=1)]
                    # else: # Don't move
                    #     actions[unit.unit_id] = [unit.recharge(0, repeat=0, n=1)]
        elif move_cost is None: # TODO: clean up w/ upper else.
            # Lets move in one of the orthogonal directions.
            other_dirs = [0,1,2,3,4]
            other_dirs.remove(direction)
            opposite_dir = opposite_directions[direction]
            if opposite_dir != 0:
                other_dirs.remove(opposite_dir)

            random_other_direction = random.choice(other_dirs)
            if self.no_collisions_next_step(unit.pos, unit, random_other_direction, game_state):
                actions[unit.unit_id] = [unit.move(random_other_direction, repeat=0, n=1)]
            else: # try moving opposite to the orthogonal direction
                if self.no_collisions_next_step(unit.pos, unit, opposite_directions[random_other_direction], game_state):
                    actions[unit.unit_id] = [unit.move(opposite_directions[random_other_direction], repeat=0, n=1)]
                # else: # Don't move
                #     actions[unit.unit_id] = [unit.recharge(0, repeat=0, n=1)]
        else:
            # pp(f"{unit.unit_id} couldn't move. {move_cost=}, {unit.action_queue_cost(game_state)=}")
            pass

    def place_factory(self, obs, game_state, metal_per_factory, water_per_factory):
        potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
        current_factories = game_state.factories[self.opp_player]
        current_fac_locations = []
        for unit_id, fac in current_factories.items():
            current_fac_locations.append(tuple(fac.pos))

        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        if len(current_fac_locations) >= 1:
            # pp(f"{len(potential_spawns)=}")
            table = np.empty((0,4))
            for spawn_choice in potential_spawns:
                ice_tile_distances = np.mean((ice_tile_locations - spawn_choice) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                fac_distances = np.mean((current_fac_locations - closest_ice_tile) ** 2, 1)
                mean_fact_dist = np.mean(fac_distances)
                # pp(f"{spawn_choice=}")
                # pp(f"{spawn_choice[0]=}, { np.min(ice_tile_distances)=}, {mean_fact_dist=}")
                row = [spawn_choice[0], spawn_choice[1], np.min(ice_tile_distances), mean_fact_dist]
                row = [int(x) for x in row]
                new_row = np.array(row).reshape(1, -1)
                # pp(f"shapes: {table.shape=}, {new_row.shape=}")
                table = np.concatenate((table, new_row), axis=0)
            
            # pp(f"{table=}")
            # spawn_arr = np.array(table)

            # sorted_indices = np.lexsort((-table[:,3], table[:,2])) 
            sorted_indices = np.lexsort((table[:,3], table[:,2])) # for min distance
            # pp(f"{sorted_indices=}")
            table = table[sorted_indices]
            # pp(f"{table[0]=}")
            ice_prox_rows = table[np.argwhere(table[:,2] == np.min(table[:,2]))]
            # pp(f"Selected top {len(ice_prox_rows)} rows: {ice_prox_rows=} ")
            # pp(f"Shapes: {table.shape=}, {ice_prox_rows.shape=}")
            # if ice_prox_rows.ndim > 2:
            #     ice_prox_rows = ice_prox_rows.squeeze() # random middle dimension added above; removing. 
            # else:
            ice_prox_rows = ice_prox_rows.reshape(-1,4)
            # pp(f"Shapes: {table.shape=}, {ice_prox_rows.shape=}")
            # far_fac_rows = ice_prox_rows[np.argwhere(ice_prox_rows[:,3] == np.max(ice_prox_rows[:,3]))]
            # for min distance
            far_fac_rows = ice_prox_rows[np.argwhere(ice_prox_rows[:,3] == np.min(ice_prox_rows[:,3]))]
            # pp(f"Selected top fac {len(far_fac_rows)} rows: {far_fac_rows=} ")

            best_spawn_choice = tuple([int(far_fac_rows[0,0,0]), int(far_fac_rows[0,0,1])])
            # pp(f"{best_spawn_choice=}")
            # First modification: lets spawn near ice.


            # print(f"INFO SHAACH Ice Tile locations: {ice_tile_locations}", file=sys.stderr)
        else:
            min_ice_dist = float(math.inf)
            best_spawn_choice = potential_spawns[np.random.randint(0, len(potential_spawns))]
            for spawn_choice in potential_spawns:
                ice_tile_distances = np.mean((ice_tile_locations - spawn_choice) ** 2, 1)
                min_ice_tile_dist = min(ice_tile_distances)
                # closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                closest_ice_tile_dist = min(ice_tile_distances)
                if closest_ice_tile_dist <= min_ice_dist:
                    best_spawn_choice = spawn_choice
                    min_ice_dist = closest_ice_tile_dist
        spawn_loc = best_spawn_choice
        return dict(spawn=spawn_loc, metal=metal_per_factory, water=water_per_factory)
    
    def next_position(self, position: np.ndarray, direction: int):
        if direction == 0:  # center
            return position
        if direction == 1:  # up
            return np.array([position[0], position[1] - 1])
        elif direction == 2:  # right
            return np.array([position[0] + 1, position[1]])
        elif direction == 3:  # down
            return np.array([position[0], position[1] + 1])
        elif direction == 4:  # left
            return np.array([position[0] - 1, position[1]])
        else:
            print(f"Error: invalid direction in next_position {direction}", file=sys.stderr)
            return position
    
    def get_next_queue_position(self, unit, action_queue):
        if len(action_queue) == 0:
            return unit.pos
        else:
            # print(f"{action_queue=}", file=sys.stderr)
            if action_queue[0][0] == 0:
                next_dir = action_queue[0][1]
                return self.next_position(unit.pos, next_dir)
            else:
                return unit.pos

    def get_manhattan_distance(self, tile_1, tile_2):
        """
        Gets the manhattan distance (grid) between 2 tiles, assumming 2d tiles.
        """
        return np.abs(tile_1[0] - tile_2[0]) + np.abs(tile_1[1]-tile_2[1])

    def get_optimal_ice_pos(self, unit_id, unit, ice_tile_locations):
        """
        Searches the full ice map, and uses target tile memory to find the
        next best target for the robot.
        """
        ice_tile_distances_map = defaultdict(list)
        ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
        closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
        target_ice_pos = closest_ice_tile #default to closest
    
        # If we already have a target, move towards it
        if unit_id in self.ice_mining_target_map:
            target_ice_pos = self.ice_mining_target_map[unit_id]
        else:
            for ice_tile in ice_tile_locations:
                ice_dist = self.get_manhattan_distance(ice_tile,unit.pos)
                # print(f"{ice_dist=}, {ice_dist.shape=}",file=sys.stderr)
                ice_tile_distances_map[ice_dist].append(ice_tile)

            sorted_distances = dict(sorted(ice_tile_distances_map.items()))
            # print(f"{sorted_distances=}",file=sys.stderr)
            for ice_dist, ice_tile in sorted_distances.items():
                target_tuple = tuple([ice_tile[0][0],ice_tile[0][1]])
                if not target_tuple in self.ice_mining_target_reverse_map:
                    # Log target
                    self.ice_mining_target_map[unit_id] = target_tuple
                    self.ice_mining_target_reverse_map[target_tuple] = unit_id
                    target_ice_pos = target_tuple
                    break
                    
        # print(f"{unit_id=}, {target_ice_pos=}",file=sys.stderr)
        return target_ice_pos
    
    def dump_ice(self, unit_id, unit, closest_factory_tile, closest_factory, actions, game_state):
        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
        direction = direction_to(unit.pos, closest_factory_tile)
        if adjacent_to_factory:
            if unit.power >= unit.action_queue_cost(game_state):
                # pp(f"ICE DUMP ACTION: {direction=},{unit.cargo.ice=}")

                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1),
                                     unit.pickup(4, 
                                                 min(closest_factory.power, 
                                                     unit.unit_cfg.BATTERY_CAPACITY - unit.power), 
                                                 repeat=0, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)    

    def fast_charge(self, unit_id, unit, closest_factory_tile, closest_factory, actions, game_state):
        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
        direction = direction_to(unit.pos, closest_factory_tile)
        if adjacent_to_factory:
            if unit.power >= unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.pickup(4, 
                                        min(closest_factory.power, 
                                            unit.unit_cfg.BATTERY_CAPACITY - unit.power), 
                                        repeat=0, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)  

    def dig_ice(self, unit_id, unit, ice_tile_locations, actions, game_state):
        target_ice_pos = self.get_optimal_ice_pos(unit=unit, unit_id=unit_id, ice_tile_locations=ice_tile_locations)
        direction = direction_to(unit.pos, target_ice_pos)
        if np.all(target_ice_pos == unit.pos):
            if len(unit.action_queue) == 0:
                if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.dig(repeat=10, n=1)]
                else:
                    pass #actions[unit_id] = [unit.recharge(x=100,repeat=0,n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)

    def get_optimal_ore_pos(self, unit_id, unit, ore_tile_locations):
        """
        Searches the full ore map, and uses target tile memory to find the
        next best target for the robot.
        """
        ore_tile_distances_map = defaultdict(list)
        ore_tile_distances = np.mean((ore_tile_locations - unit.pos) ** 2, 1)
        closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
        target_ore_pos = closest_ore_tile #default to closest
    
        # If we already have a target, move towards it
        if unit_id in self.ore_mining_target_map:
            target_ore_pos = self.ore_mining_target_map[unit_id]
        else:
            for ore_tile in ore_tile_locations:
                ore_dist = self.get_manhattan_distance(ore_tile,unit.pos)
                # print(f"{ice_dist=}, {ice_dist.shape=}",file=sys.stderr)
                ore_tile_distances_map[ore_dist].append(ore_tile)

            sorted_distances = dict(sorted(ore_tile_distances_map.items()))
            # print(f"{sorted_distances=}",file=sys.stderr)
            for ore_dist, ore_tile in sorted_distances.items():
                target_tuple = tuple([ore_tile[0][0],ore_tile[0][1]])
                if not target_tuple in self.ore_mining_target_reverse_map:
                    # Log target
                    self.ore_mining_target_map[unit_id] = target_tuple
                    self.ore_mining_target_reverse_map[target_tuple] = unit_id
                    target_ore_pos = target_tuple
                    break
        # print(f"{unit_id=}, {target_ice_pos=}",file=sys.stderr)
        return target_ore_pos
    
    def dump_ore(self, unit_id, unit, closest_factory_tile, closest_factory, actions, game_state):
        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
        direction = direction_to(unit.pos, closest_factory_tile)
        if adjacent_to_factory:
            if unit.power >= unit.action_queue_cost(game_state) and unit_id not in actions:
                # resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
                # pp(f"ORE DUMP ACTION: {direction=},{unit.cargo.ore=}")
                actions[unit_id] = [unit.transfer(direction, 1, unit.cargo.ore, repeat=0, n=1),
                                     unit.pickup(4, 
                                                 min(closest_factory.power, 
                                                     unit.unit_cfg.BATTERY_CAPACITY - unit.power), 
                                                 repeat=0, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)    

    def dig_ore(self, unit_id, unit, ore_tile_locations, actions, game_state):
        target_ore_pos = self.get_optimal_ore_pos(unit=unit, unit_id=unit_id, ore_tile_locations=ore_tile_locations)
        direction = direction_to(unit.pos, target_ore_pos)
        if np.all(target_ore_pos == unit.pos):
            if len(unit.action_queue) == 0:
                if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.dig(repeat=10, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)

    def get_optimal_hit_pos(self,unit, unit_id, target_lichen_tiles, onboard_lichen_map):
        """
        Searches the full lichen map, and uses target tile memory to find the
        next best target for the robot.
        """
        lichen_tile_distances_map = defaultdict(list)
        lichen_tile_distances = np.mean((target_lichen_tiles - unit.pos) ** 2, 1)
        closest_lichen_tile = target_lichen_tiles[np.argmin(lichen_tile_distances)]
        target_lichen_pos = closest_lichen_tile #default to closest
    
        # If we already have a target, move towards it
        # pp(f"{target_lichen_pos=}")
        # pp(f"{onboard_lichen_map[tuple(target_lichen_pos)]=}")
        if unit_id in self.attack_target_map:
            
            recorded_lichen_pos = self.attack_target_map[unit_id]
            if onboard_lichen_map[tuple(recorded_lichen_pos)] > 0:
                target_lichen_pos = recorded_lichen_pos
                # pp(f"IF {target_lichen_pos=}")
                # pp(f"IF {onboard_lichen_map[tuple(target_lichen_pos)]=}")
            else:
                del self.attack_target_map[unit_id]

        if unit_id not in self.attack_target_map:
            for lichen_tile in target_lichen_tiles:
                lichen_dist = self.get_manhattan_distance(lichen_tile,unit.pos)
                # print(f"{ice_dist=}, {ice_dist.shape=}",file=sys.stderr)
                lichen_tile_distances_map[lichen_dist].append(lichen_tile)

            sorted_distances = dict(sorted(lichen_tile_distances_map.items()))
            # print(f"{sorted_distances=}",file=sys.stderr)
            for lichen_dist, lichen_tile in sorted_distances.items():
                target_tuple = tuple([lichen_tile[0][0],lichen_tile[0][1]])
                if not target_tuple in self.attack_target_reverse_map:
                    # Log target
                    self.attack_target_map[unit_id] = target_tuple
                    self.attack_target_reverse_map[target_tuple] = unit_id
                    target_lichen_pos = target_tuple
                    # pp(f"ELSE {target_lichen_pos=}")
                    # pp(f"ELSE {onboard_lichen_map[tuple(target_lichen_pos)]=}")
                    break
                    
        # print(f"{unit_id=}, {target_ice_pos=}",file=sys.stderr)
        return target_lichen_pos        


    def attack_lichen(self, unit_id, unit, target_lichen_tiles, actions, game_state):
        onboard_lichen_map = game_state.board.lichen
        # if game_state.real_env_steps == 902:
        #     i=0
        #     for row in onboard_lichen_map:
        #         pp(f"{i}{row}")
        #         i+=1
        target_lichen_pos = self.get_optimal_hit_pos(unit=unit, unit_id=unit_id, 
                                                     target_lichen_tiles=target_lichen_tiles,
                                                     onboard_lichen_map=onboard_lichen_map)
        # if onboard_lichen_map[tuple(target_lichen_pos)] <= 0:
        #     pp(f"ALERT! Bad target assigned to {unit_id=}")
        direction = direction_to(unit.pos, target_lichen_pos)
        # pp(f"Target lichen pos = {target_lichen_pos} lichen on it: {onboard_lichen_map[tuple(target_lichen_pos)]} {unit_id=}, {direction=}")
        if np.all(target_lichen_pos == unit.pos):
            if len(unit.action_queue) == 0:# and onboard_lichen_map[tuple(target_lichen_pos)] > 0:
                # if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                repeats = math.ceil(onboard_lichen_map[tuple(target_lichen_pos)] / 10)
                # pp(f"Lichen dig repeats = {repeats}")
                actions[unit_id] = [unit.dig(repeat=repeats, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)


    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        original_factories_to_place = 0
        # print(f"{step=}",file=sys.stderr) 
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # Initialize
            if self.planned_factories_to_place == 0:
                original_factories_to_place = game_state.teams[self.player].factories_to_place
                # self.planned_factories_to_place = max(original_factories_to_place // 2, 1)
                self.planned_factories_to_place =  original_factories_to_place
                # how much water and metal you have in your starting pool to give to new factories
                start_water_left = game_state.teams[self.player].water
                start_metal_left = game_state.teams[self.player].metal
                self.water_per_factory = math.ceil(start_water_left / self.planned_factories_to_place)
                self.metal_per_factory = math.ceil(start_metal_left / self.planned_factories_to_place)
                self.total_original_water = start_water_left
            # factory placement period
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            print(f"{factories_to_place=}, \
                   {self.planned_factories_to_place=}, \
                   {self.water_per_factory=} \
                    {self.metal_per_factory=}", file=sys.stderr)

            # if factories_to_place >= self.planned_factories_to_place and my_turn_to_place:
            if factories_to_place <= self.planned_factories_to_place and my_turn_to_place:
                water_left = game_state.teams[self.player].water
                metal_left = game_state.teams[self.player].metal 
                m = self.metal_per_factory
                w = self.water_per_factory 
                if self.metal_per_factory > metal_left:
                    m = metal_left
                if self.water_per_factory > water_left:
                    w = water_left
                return self.place_factory(obs, game_state, m, w)
            return dict()

    def nearest_rubble_loc(self, unit, rubble_locations):         
        #print(f"Rubble locations shape {rubble_locations.shape}", file=sys.stderr)
        rubble_tile_distances = np.mean((rubble_locations - unit.pos) ** 2, 1)
        closest_rubble_tile = rubble_locations[np.argmin(rubble_tile_distances)]
        #print(f" SHAACH 14 {closest_rubble_tile=}", file=sys.stderr)
        return closest_rubble_tile

    def cost_of_reaching_closest_factory(self, current_loc, factory_loc):
        return self.get_manhattan_distance(current_loc, factory_loc)

    # logic for light robots to clear the rubel
    def dig_rubble(self, unit, closest_factory_tile, closest_factory, actions, game_state):
        """
        rubble_amount = 85
        dig_capacity = 20   -> n = dig_capacity
        dig_capacity = 50   -> n = rubble_amount / 2 
        """

        rubble_locs = np.argwhere(game_state.board.rubble >=1)
        # print(f"Rubble Amount at location at 63, 60 {game_state.board.rubble[63, 60]}", file=sys.stderr)
        # print(f"Rubble locations {rubble_locs}, Max rubble {np.max(game_state.board.rubble)}, Min rubble {np.min(game_state.board.rubble)}", file=sys.stderr)

        starting_loc = self.nearest_rubble_loc(unit, rubble_locs)
        # print(f"{starting_loc=}, {type(starting_loc)}", file=sys.stderr)
        power_to_save = self.cost_of_reaching_closest_factory(closest_factory_tile, unit.pos) #+ 20
        dig_capacity = np.max((unit.power - power_to_save) // 5, 0)
        rubble_amount = game_state.board.rubble[starting_loc.item(0), starting_loc.item(1)]
        if dig_capacity > 0:
            n1 = 0
            repeat1 = 0
            if np.all(starting_loc == unit.pos):

                # if dig_capacity > 0:
                # Decide how much to dig in this and the next turn
                if rubble_amount <= (2 * dig_capacity):
                    n1 = math.ceil(rubble_amount / 2)
                    repeat1 = 0
                else:
                    n1 = dig_capacity
                    repeat1 = math.ceil((rubble_amount - (dig_capacity * 2)) / 2)

                # print(f"SHAACH 15 {dig_capacity=} {rubble_amount=}", file=sys.stderr)
                # print(f"SHAACH 15 {n1=} {repeat1=}", file=sys.stderr)
                # if n > 0:
                if unit.unit_id in actions:
                    actions[unit.unit_id].append(unit.dig(repeat=repeat1, n=n1))
                else:
                    actions[unit.unit_id] = [unit.dig(repeat=repeat1, n=n1)]

            else:
                direction = direction_to(unit.pos, starting_loc)
                self.move_bot(direction, unit, game_state, actions)

        else: # dig capacity is not enough, need to go to the factory
            adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
            
            if adjacent_to_factory:
                if unit.unit_id not in actions:
                    actions[unit.unit_id] = [unit.pickup(4, min(closest_factory.power, unit.unit_cfg.BATTERY_CAPACITY - unit.power), repeat=0, n=1)]
                else:
                    actions[unit.unit_id].append(unit.pickup(4, min(closest_factory.power, unit.unit_cfg.BATTERY_CAPACITY - unit.power), repeat=0, n=1))
            else:
                direction = direction_to(unit.pos, closest_factory_tile)
                self.move_bot(direction, unit, game_state, actions)

    def build_graph(self, game_state):
        """
        Let factory tiles, and resource tiles have infinite weight, so the 
        robot skips over those.

        """
        board = game_state.board.rubble
        opp_factories = game_state.factories[self.opp_player]
        ice_tiles = np.argwhere(game_state.board.ice==1)
        pp(f"{ice_tiles=}")
        for (i,j) in ice_tiles:
                board[i][j] = 9999
        def get_full_fac(pos):
            fac = []
            for i in range(pos[0]-1,pos[0]+2):
                for j in range(pos[1]-1,pos[1]+2):
                    fac.append((i,j))
            return fac

        for unit_id, unit in opp_factories.items():
            # pp(f"FAC {unit.pos=}")
            full_fac = get_full_fac(unit.pos)
            # pp(f"FAC full = {full_fac}")
            for (i,j) in full_fac:
                board[i][j] = 9999

        # for row in range(0,board.shape[0]):
        #     # for col in range(0, board.shape[1]):
        #     #     pp(f"{board[row][col]} ",sep =' ')
        #     pp(f"{board[row]}")

        adj_mat = np.zeros((4096,4096))

        def get_news(i):
            north = None if i<=63 else i-64
            south = None if i>=4032 else i+64
            east = None if i%64==63 else i+1
            west = None if i%64==0 else i-1
            return north, south, east, west
        for i in range(4096):
            north, south, east, west = get_news(i)
            if north or north == 0:
                adj_mat[i][north] = 9999 if i<=63 else  max(board[i//64 - 1][i%64], 1)
            if south or south == 0:
                adj_mat[i][south] = 9999 if i>=4032 else max(board[i//64 + 1][i%64], 1)
            if east or east == 0: 
                adj_mat[i][east] = 9999 if i%64==63 else max(board[i//64][i%64+1], 1)
            if west or west == 0:
                adj_mat[i][west] = 9999 if i%64==0 else max(board[i//64][i%64-1], 1)
            if i == 4095 or i == 4031:
                pp(f"{i=}")
                pp(f"{north=},{south=},{east=},{west=}")
                if north or north == 0:
                    pp(f"{adj_mat[i][north]=}")
                if south or south == 0:
                    pp(f"{adj_mat[i][south]=}")
                if east or east == 0:
                    pp(f"{adj_mat[i][east]=}")
                if west or west == 0:
                    pp(f"{adj_mat[i][west]=}")
        pp(f"{board[62]=}")
        pp(f"{board[63]=}")
        # pp(f"{adj_mat[0][0]=}, {adj_mat[0][1]=}, {adj_mat[0][64]=}")
        # pp(f"{adj_mat[1][0]=}, {adj_mat[1][2]=}, {adj_mat[1][65]=}")
        # pp(f"{adj_mat[64]=}")
        # pp(f"{adj_mat[4032]=}")
        # pp(f"{adj_mat[4095]=}")

        G = nx.from_numpy_array(adj_mat,edge_attr="weight")
        pp(f"{type(G.edges(data=True))=}")
        pp(f"{dir(G.edges(data=True))=}")
        pp(f"{G.number_of_nodes()=}")
        pp(f"All edges with key 0:{[(i, j)   for i, j in G.edges if i == 0]}")
        pp(f"All nodes connected to edges with key 0: {set( [n for i, j in G.edges if i == 0  for n in [j]] )}")
        pp(f"All edges with key 1:{[(i, j)   for i, j in G.edges if i == 1]}")
        pp(f"All nodes connected to edges with key 1: {set( [n for i, j in G.edges if i == 1  for n in [j]] )}")
        pp(f"All edges with key 4095:{[(i, j)   for i, j in G.edges if i == 4095]}")
        pp(f"All nodes connected to edges with key 4095: {set( [n for i, j in G.edges if i == 4095  for n in [j]] )}")
        pp(f"All edges with key 4096:{[(i, j)   for i, j in G.edges if i == 4096]}")
        pp(f"All nodes connected to edges with key 4096: {set( [n for i, j in G.edges if i == 4096  for n in [j]] )}")
        return G

    def generate_path(self, G, src, dest):
        pp(f"Looking for a path from {src} to {dest}")
        start = src[0] * 64 + src[1]%64
        stop = dest[0] * 64 + dest[1]%64

        pp(f"All edges with key {start}:{[(i, j)   for i, j in G.edges if i == start]}")
        pp(f"All nodes connected to edges with key {start}: {set( [n for i, j in G.edges if i == start  for n in [j]] )}")
        
        pp(f"All edges with key {stop}:{[(i, j)   for i, j in G.edges if i == stop]}")
        pp(f"All nodes connected to edges with key {stop}: {set( [n for i, j in G.edges if i == stop  for n in [j]] )}")



        path = nx.shortest_path(G, start, stop, weight="weight")
        tile_path = [(p//64, p%64) for p in path]
        pp(f"Path from {src} ({start}) to {dest} ({stop}) = {path}")
        pp(f"Path from {src} ({start}) to {dest} ({stop}) = {tile_path}")
        return tile_path

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        
        HEAVY_CARGO_TARGET = 160
        LIGHT_CARGO_TARGET = 100
        LICHEN_PROD_STEPS = 700
        # self.ATTACK_MODE = False
        LICHEN_ATTACK_RADIUS = 64 #full board
        AGGRESSION_DIVISOR = 1  #max aggressive = 1, least = #light bots
        WATER_GONE_TOL = 1.00 # 10% tolerance for pred vs actual

        if game_state.real_env_steps == 1:
            pp(f"Build graph, steps = {game_state.real_env_steps}")
            G = self.build_graph(game_state=game_state)
            
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if game_state.real_env_steps == 1:
                ore_map = game_state.board.ore
                ore_tile_locations = np.argwhere(ore_map == 1)
                ore_tile_distances = np.mean((ore_tile_locations - factory.pos) ** 2, 1)
                closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
                self.generate_path(G, tuple(factory.pos),tuple(closest_ore_tile))

            # Try building heavy first. If not enough power, build light.
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                actions[unit_id] = factory.build_light()
            # Lichen
            # if self.env_cfg.max_episode_length - game_state.real_env_steps < LICHEN_PROD_STEPS:
            #     if factory.water_cost(game_state) <= factory.cargo.water:
            #         actions[unit_id] = factory.water()
            if  game_state.real_env_steps > LICHEN_PROD_STEPS: 
                remaining_turns = self.env_cfg.max_episode_length - game_state.real_env_steps
                factory_total_water_production = game_state.real_env_steps - 150
                water_pred_end_game = factory.cargo.water + \
                                        (remaining_turns * factory_total_water_production / game_state.real_env_steps) - \
                                        (remaining_turns + 1) * factory.water_cost(game_state) * WATER_GONE_TOL
                # pp(f"{unit_id=} {water_pred_end_game=}")
                if water_pred_end_game > 0:
                    # if factory.water_cost(game_state) <= factory.cargo.water:
                        actions[unit_id] = factory.water()

            # Halfway through the game, start checking on opponent
            if self.env_cfg.max_episode_length - game_state.real_env_steps < 500: 

                units = game_state.units[self.player]
                opp_factories = game_state.factories[self.opp_player]
                onboard_lichen_map = game_state.board.lichen

                onboard_lichen_tiles = np.argwhere(onboard_lichen_map > 0)

                # print(f"{onboard_lichen_tiles=}",file=sys.stderr)
                # print(f"{game_state.board.lichen_strains =}",file=sys.stderr)
                # print(f"{factory.strain_id=}",file=sys.stderr)

                if len(onboard_lichen_tiles) > 0:
                    # Find targets
                    target_lichen_tiles = []
                    for fac_id, fac in opp_factories.items():
                        fac_lichens = np.argwhere(game_state.board.lichen_strains == fac.strain_id)
                        # target_lichen_tiles.extend(fac_lichens)
                        # pp(f"{fac.strain_id=}")
                        # pp(f"{fac_lichens[0][0:3]=}")
                        # pp(f"{onboard_lichen_map[tuple(fac_lichens[0][0:3])]=}")
                        # pp(f"{game_state.board.lichen_strains[tuple(fac_lichens[0][0:3])]=}")
                        for lichen in fac_lichens:
                            lic_fac_dis = self.get_manhattan_distance(lichen, fac.pos)
                            if lic_fac_dis <= LICHEN_ATTACK_RADIUS:
                                target_lichen_tiles.append(lichen)

                    # Enlist kill bots
                    light_units = dict() # unit_id, pos
                    for unit_id, unit in units.items():
                        if unit.unit_type == "LIGHT":
                            light_units[unit_id] = unit.pos
                    available_attack_bots = len(light_units) // AGGRESSION_DIVISOR
                    # if available_attack_bots > 0 and 
                    if len(target_lichen_tiles) > 0:
                        self.ATTACK_MODE = True
                        # print(f"boom {self.ATTACK_MODE=}",file=sys.stderr)
                        # pp(f"{target_lichen_tiles=}")
                        self.bot_role["redbot"] = []
                        light_keys = list(light_units.keys())
                        for i in range(available_attack_bots):
                            self.bot_role["redbot"].append(light_keys[i])
                        self.bot_role["builder"] = []
                        for i in range(available_attack_bots,len(light_units)):
                            self.bot_role["builder"].append(light_keys[i])
                            i+=1
                        # pp(f"{self.bot_role}")
                else:
                    self.ATTACK_MODE = False  

            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = game_state.board.ore
        ore_tile_locations = np.argwhere(ore_map == 1)
        
        red_bots = set(self.bot_role["redbot"]) if self.ATTACK_MODE else []
        build_bots = set(self.bot_role["builder"]) if self.ATTACK_MODE else []
        for unit_id, unit in units.items():
            # track the closest factory
            closest_factory = None
            # adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                # adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 4

                if unit.unit_type=="HEAVY":
                    if unit.cargo.ice < HEAVY_CARGO_TARGET:
                        self.dig_ice(unit_id, unit, ice_tile_locations, actions, game_state)
                    # else if we have enough ice, we go back to the factory and dump it.
                    elif unit.cargo.ice >= HEAVY_CARGO_TARGET:
                        self.dump_ice(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)

                elif unit.unit_type=="LIGHT": 
                    #print(f"Rubbles {game_state.board.rubble}", file=sys.stderr)
                    if self.ATTACK_MODE and len(target_lichen_tiles) > 0:# and unit.unit_id in red_bots:
                        if unit.cargo.ore > 0:
                            self.dump_ore(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)
                        else:
                            if unit.power > 10: # Todo: fix - bot can get stuck around 10, aiming alternately for if/else tiles.
                                self.attack_lichen(unit_id, unit, target_lichen_tiles, actions, game_state)
                            else:
                                self.fast_charge(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)
                    else:#if unit.unit_id in build_bots:
                        # if game_state.real_env_steps < 250:
                        if int(re.findall(r'\d+',unit_id)[0])%3 == 0:
                            if unit.cargo.ore > 0:
                                self.dump_ore(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state) 
                            else:
                                self.dig_rubble(unit, closest_factory_tile, closest_factory, actions, game_state)
                        else:
                            if unit.cargo.ore < LIGHT_CARGO_TARGET:
                                self.dig_ore(unit_id, unit, ore_tile_locations, actions, game_state)
                            # else if we have enough ice, we go back to the factory and dump it.
                            elif unit.cargo.ore >= LIGHT_CARGO_TARGET:
                                self.dump_ore(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)
                        # else: # First focus on building more bots
                        #         if unit.cargo.ore < LIGHT_CARGO_TARGET:
                        #             self.dig_ore(unit_id, unit, ore_tile_locations, actions, game_state)
                        #         # else if we have enough ice, we go back to the factory and dump it.
                        #         elif unit.cargo.ore >= LIGHT_CARGO_TARGET:
                        #             self.dump_ore(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)
        return actions
    



"""
Attack mode

Destroy enemy lichen farms. 

1. Detect that enemy has started growing Lichen.
2. Take n = #enemy factory light bots, and fully charge them - mark as kill bots. [bot role map]
3. Detect target lichen location: closer to the enemy factory, the better. [lichen target map]
4. Dispatch kill bots to target locations. 
"""