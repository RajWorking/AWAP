import random
from collections import deque
from typing import Tuple, Optional, List

from game_constants import Team, TileType, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food

class BotPlayer:
    """
    Duo Noodle Bot (generalized)
    --------------------------------
    This bot is a cleaned-up and generalized version of the original duo_noodle_bot.
    It is designed to fulfill ANY active order it finds (first active order), rather
    than a hard-coded meat+noodles recipe.

    High-level strategy:
     - Find an active order and iterate through its required ingredients.
     - For each required ingredient:
         * Buy the ingredient at a shop.
         * If the ingredient can immediately be cooked (controller.can_start_cook),
           start cooking on the cooker (ensuring a pan exists on the cooker).
         * Else, place on the nearest counter and attempt to chop (if applicable),
           then pick up.
         * If the item was cooked on a pan, wait until it's cooked (cooked_stage == 1),
           then take it from the pan.
         * Add the obtained ingredient to the plate on the assembly counter.
     - After all items are on a plate, pick up the plate and submit.
     - If something goes wrong (e.g., burnt food or bot is holding garbage), trash and
       restart the order.
    Notes:
     - This bot is conservative and uses the controller's CAN_* and action return values
       to detect whether an action is valid/succeeded, so it should adapt to different
       food processing rules without hard-coding which foods are choppable/cookable.
     - The implementation keeps the original random movement for the second bot so the
       team still moves both bots each turn.
    """

    def __init__(self, map_copy):
        self.map = map_copy
        self.assembly_counter = None
        self.cooker_loc = None
        self.my_bot_id = None

        # State machine state
        self.state = 0

        # Order / assembly tracking (generalized)
        self.current_order_id = None
        self.order_requirements: List[str] = []  # list of food names (strings) from order
        self.item_idx = 0  # index into order_requirements for which ingredient we're working on
        self.plate_on_counter = False  # whether we placed a clean plate on the assembly counter
        self.waiting_for_cook = False  # waiting for cooker to finish current cooking

    def get_bfs_path(self, controller: RobotController, start: Tuple[int, int], target_predicate) -> Optional[Tuple[int, int]]:
        queue = deque([(start, [])])
        visited = set([start])
        w, h = self.map.width, self.map.height

        while queue:
            (curr_x, curr_y), path = queue.popleft()
            tile = controller.get_tile(controller.get_team(), curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                if not path: return (0, 0)
                return path[0]

            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def move_towards(self, controller: RobotController, bot_id: int, target_x: int, target_y: int) -> bool:
        """
        Move the bot one step towards being adjacent to (target_x, target_y).
        Returns True if already adjacent (or at target) and no movement needed.
        Otherwise performs one move toward the target (if possible) and returns False.
        """
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state['x'], bot_state['y']

        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1

        if is_adjacent_to_target(bx, by, None): return True
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target)
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            return False
        return False

    def find_nearest_tile(self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str) -> Optional[Tuple[int, int]]:
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == tile_name:
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos

    def pick_order(self, controller: RobotController) -> bool:
        """
        Choose an active order to fulfill. This picks the first active order returned by the controller.
        Initializes order_requirements (list of strings). Returns True if an order was selected.
        """
        orders = controller.get_orders()
        for o in orders:
            if o.get('is_active'):
                reqs = o.get('required') or []
                # Some maps return names or FoodType objects; normalize to strings
                normalized = []
                for r in reqs:
                    if isinstance(r, str):
                        normalized.append(r)
                    else:
                        # Try to extract a string if it's a dict-like public item representation
                        try:
                            normalized.append(str(r))
                        except Exception:
                            pass
                if not normalized:
                    continue
                self.current_order_id = o.get('order_id')
                self.order_requirements = normalized
                self.item_idx = 0
                self.plate_on_counter = False
                self.waiting_for_cook = False
                return True
        return False

    def play_turn(self, controller: RobotController):
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots: return

        self.my_bot_id = my_bots[0]
        bot_id = self.my_bot_id

        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']

        if self.assembly_counter is None:
            self.assembly_counter = self.find_nearest_tile(controller, bx, by, "COUNTER")
        if self.cooker_loc is None:
            self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")

        if not self.assembly_counter or not self.cooker_loc: return

        cx, cy = self.assembly_counter
        kx, ky = self.cooker_loc

        # If we're holding anything unexpectedly in certain states, go trash it and restart
        if self.state in [2, 8, 10] and bot_info.get('holding'):
            self.state = 16

        # State 0: initialization - ensure a pan is on the cooker or buy one
        if self.state == 0:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(tile.item, Pan):
                self.state = 2
            else:
                self.state = 1

        # State 1: buy a pan and place it on cooker
        elif self.state == 1:
            holding = bot_info.get('holding')
            if holding:
                # We assume it's the pan we're holding; move it to cooker and place
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        self.state = 2
            else:
                shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
                if not shop_pos: return
                sx, sy = shop_pos
                if self.move_towards(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        # State 2: select an order to fulfill (generalized)
        elif self.state == 2:
            if self.pick_order(controller):
                # prepare to assemble a plate for this order: buy a plate first
                self.state = 8  # move to plate-buying state
            else:
                # no active orders - idle / gather money or wait
                return

        # State 8: buy the plate for assembly
        elif self.state == 8:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos: return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                # We prefer buying a plate rather than taking from sink; buy if affordable
                if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        self.state = 9

        # State 9: place the plate on the assembly counter
        elif self.state == 9:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.plate_on_counter = True
                    self.state = 10

        # State 10: For each ingredient in the order, process and add to plate
        elif self.state == 10:
            # If we've finished all ingredients, pickup plate and submit
            if self.item_idx >= len(self.order_requirements):
                self.state = 14  # pick up plate
                return

            # Work on the next required ingredient
            req_name = self.order_requirements[self.item_idx]

            # Normalize possible formatting (e.g., FoodType enum values may be used)
            if isinstance(req_name, str):
                req_name_up = req_name.upper()
            else:
                req_name_up = str(req_name).upper()

            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos: return
            sx, sy = shop_pos

            # If we're not holding the ingredient, buy it
            if not bot_info.get('holding'):
                if self.move_towards(controller, bot_id, sx, sy):
                    # Try to buy by mapping the name to FoodType attribute, falling back to string
                    ft = None
                    try:
                        ft = getattr(FoodType, req_name_up)
                    except Exception:
                        ft = None
                    # If mapping succeeded and team has money, buy
                    if ft is not None and controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                        if controller.buy(bot_id, ft, sx, sy):
                            # after buying, next tick we'll attempt to process it
                            return
                    else:
                        # If we couldn't map to FoodType, try to skip (defensive)
                        # Reset order to avoid infinite stuck
                        self.current_order_id = None
                        self.order_requirements = []
                        self.state = 2
                        return
                return

            # At this point, bot is holding the raw ingredient. Decide where to put/process it.
            # 1) If we can start cook on cooker, do so (this implicitly requires a pan on the cooker)
            if controller.can_start_cook(bot_id, kx, ky):
                if self.move_towards(controller, bot_id, kx, ky):
                    # Start the cook; start_cook returns True if cooking begins
                    if controller.start_cook(bot_id, kx, ky):
                        # mark that we're waiting for this cook to finish
                        self.waiting_for_cook = True
                        # After placing the food to cook, we want to continue onto the next ingredient
                        # only when the cooked product is ready and taken from pan.
                        # Keep item_idx the same until we take the cooked food.
                        return
                return

            # 2) Otherwise, attempt to place on assembly counter and chop if necessary, then pickup
            if self.move_towards(controller, bot_id, cx, cy):
                # Try placing onto counter
                if controller.place(bot_id, cx, cy):
                    # Try chopping (controller.chop returns True if a chop action happened)
                    if controller.chop(bot_id, cx, cy):
                        # after chopping, try to pick up
                        if controller.pickup(bot_id, cx, cy):
                            # done processing this ingredient, move to putting on cooker / plate as needed
                            # if the ingredient is cookable, we'll handle that in the next iteration (we're holding it now)
                            # else, add to plate if possible
                            # Attempt to add immediately to plate on counter
                            if controller.add_food_to_plate(bot_id, cx, cy):
                                # successful add to plate; move to next ingredient
                                self.item_idx += 1
                                return
                            else:
                                # couldn't add directly to plate (maybe plate missing) -> keep item in hand to place later
                                return
                    else:
                        # Chop didn't happen (maybe not choppable). Try to pick up immediately in case it was placed but doesn't need chopping
                        if controller.pickup(bot_id, cx, cy):
                            # after pickup, try to add to plate
                            if controller.add_food_to_plate(bot_id, cx, cy):
                                self.item_idx += 1
                                return
                            return
                return

        # State 12: If we had started cooking, wait by checking cooker for cooked food and take it when ready
        elif self.state == 12:
            # This state was used in older logic; keep for compatibility but prefer the dynamic approach above.
            self.state = 10

        # State 13: add meat to plate (kept for readability; dynamic code in state 10 handles adds)
        elif self.state == 13:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = 14

        # State 14: pick up the plate from counter (once all ingredients added)
        elif self.state == 14:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = 15

        # State 15: submit the plate at SUBMIT
        elif self.state == 15:
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            if not submit_pos:
                return
            ux, uy = submit_pos
            if self.move_towards(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    # reset order tracking and go pick the next order
                    self.current_order_id = None
                    self.order_requirements = []
                    self.item_idx = 0
                    self.plate_on_counter = False
                    self.waiting_for_cook = False
                    self.state = 2

        # State 16: trash currently held item and restart the order selection
        elif self.state == 16:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos: return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty):
                if controller.trash(bot_id, tx, ty):
                    # restart order selection
                    self.current_order_id = None
                    self.order_requirements = []
                    self.item_idx = 0
                    self.plate_on_counter = False
                    self.waiting_for_cook = False
                    self.state = 2

        # After attempting to handle the primary bot's actions, also move other bots a bit to ensure they act
        for i in range(1, len(my_bots)):
            self.my_bot_id = my_bots[i]
            bot_id = self.my_bot_id

            bot_info = controller.get_bot_state(bot_id)
            bx, by = bot_info['x'], bot_info['y']

            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            nx,ny = bx + dx, by + dy
            if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                controller.move(bot_id, dx, dy)
                return
