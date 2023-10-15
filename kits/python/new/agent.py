from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import math, random
from collections import defaultdict
import sys

move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
opposite_directions = {1:3,3:1,2:4,4:2,0:0}

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.planned_factories_to_place = 0
        self.metal_per_factory = 150
        self.water_per_factory = 150
        self.mining_target_map = dict() # dict to map robot to its mining target
        self.mining_target_reverse_map = dict() # dict to map mining target tile to robot id
        self.total_original_water = 0
        self.DROUGHT_STATE = False
        self.DROUGHT_RECOVERY_COUNTER = 10

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
            # if robot.pos[0] == target_pos[0] and robot.pos[1] == target_pos[1]:
            #     return False
            next_pos = self.get_next_queue_position(robot, robot.action_queue)
            if next_pos[0] == target_pos[0] and next_pos[1] == target_pos[1]:
                return False
        for robot_id, robot in op_robots.items():
            next_pos = self.get_next_queue_position(robot, robot.action_queue)
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
            if self.no_collisions_next_step(unit.pos, direction, game_state):
                actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]
            else:
                # Lets move in one of the orthogonal directions.
                other_dirs = [0,1,2,3,4]
                other_dirs.remove(direction)

                opposite_dir = opposite_directions[direction]
                if opposite_dir != 0:
                    other_dirs.remove(opposite_dir)

                random_other_direction = random.choice(other_dirs)
                if self.no_collisions_next_step(unit.pos, random_other_direction, game_state):
                    actions[unit.unit_id] = [unit.move(random_other_direction, repeat=0, n=1)]
                else: # try moving opposite to the orthogonal direction
                    if self.no_collisions_next_step(unit.pos, opposite_directions[random_other_direction], game_state):
                        actions[unit.unit_id] = [unit.move(opposite_directions[random_other_direction], repeat=0, n=1)]
                    # else: # Don't move
                    #     actions[unit.unit_id] = [unit.recharge(0, repeat=0, n=1)]

    def place_factory(self, obs, game_state, metal_per_factory, water_per_factory):
        potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

        # First modification: lets spawn near ice.
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        # print(f"INFO SHAACH Ice Tile locations: {ice_tile_locations}", file=sys.stderr)

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
        if unit_id in self.mining_target_map:
            target_ice_pos = self.mining_target_map[unit_id]
        else:
            for ice_tile in ice_tile_locations:
                ice_dist = self.get_manhattan_distance(ice_tile,unit.pos)
                # print(f"{ice_dist=}, {ice_dist.shape=}",file=sys.stderr)
                ice_tile_distances_map[ice_dist].append(ice_tile)

            sorted_distances = dict(sorted(ice_tile_distances_map.items()))
            # print(f"{sorted_distances=}",file=sys.stderr)
            for ice_dist, ice_tile in sorted_distances.items():
                target_tuple = tuple([ice_tile[0][0],ice_tile[0][1]])
                if not target_tuple in self.mining_target_reverse_map:
                    # Log target
                    self.mining_target_map[unit_id] = target_tuple
                    self.mining_target_reverse_map[target_tuple] = unit_id
                    target_ice_pos = target_tuple
                    break
                    
        # print(f"{unit_id=}, {target_ice_pos=}",file=sys.stderr)
        return target_ice_pos
    
    def dump_ice(self, unit_id, unit, closest_factory_tile, closest_factory, actions, game_state):
        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
        direction = direction_to(unit.pos, closest_factory_tile)
        if adjacent_to_factory:
            if unit.power >= unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1),
                                     unit.pickup(4, 
                                                 min(closest_factory.power, 
                                                     unit.unit_cfg.BATTERY_CAPACITY - unit.power), 
                                                 repeat=0, n=1)]
        else:
            self.move_bot(direction, unit, game_state, actions)    

    def dig_ice(self, unit_id, unit, ice_tile_locations, actions, game_state):
        target_ice_pos = self.get_optimal_ice_pos(unit=unit, unit_id=unit_id, ice_tile_locations=ice_tile_locations)
        if np.all(target_ice_pos == unit.pos):
            if len(unit.action_queue) == 0:
                if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.dig(repeat=10, n=1)]
                else:
                    pass #actions[unit_id] = [unit.recharge(x=100,repeat=0,n=1)]
        else:
            direction = direction_to(unit.pos, target_ice_pos)
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


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        
        HEAVY_CARGO_TARGET = 160
        LIGHT_CARGO_TARGET = 50
        LICHEN_PROD_STEPS = 200

        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            # Try building heavy first. If not enough power, build light.
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                actions[unit_id] = factory.build_light()
            # Lichen
            if self.env_cfg.max_episode_length - game_state.real_env_steps < LICHEN_PROD_STEPS:
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
                    if self.env_cfg.max_episode_length - game_state.real_env_steps > LICHEN_PROD_STEPS:
                        self.dig_rubble(unit, closest_factory_tile, closest_factory, actions, game_state)
                    else:
                        if unit.cargo.ice < LIGHT_CARGO_TARGET:
                            self.dig_ice(unit_id, unit, ice_tile_locations, actions, game_state)
                        # else if we have enough ice, we go back to the factory and dump it.
                        elif unit.cargo.ice >= LIGHT_CARGO_TARGET:
                            self.dump_ice(unit_id, unit, closest_factory_tile, closest_factory, actions, game_state)
        return actions
