"""
Command-based bot framework for Overcooked-style game.

This bot uses a command queue system where each command represents an atomic action
(buy, chop, cook, assemble, submit). Orders are converted into command sequences
that the bot executes sequentially.
"""

from collections import deque
from typing import Tuple, Optional, List, Dict
from abc import ABC, abstractmethod

from robot_controller import RobotController
from game_constants import FoodType, ShopCosts


# ============================================================================
# PATHFINDING UTILITIES
# ============================================================================

def bfs_to_adjacent(controller: RobotController, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """BFS pathfinding: returns next move (dx, dy) toward goal. Returns None if adjacent or unreachable."""
    gx, gy = goal
    sx, sy = start

    # Already adjacent (Chebyshev distance <= 1)?
    if max(abs(sx - gx), abs(sy - gy)) <= 1:
        return None

    queue = deque([(start, [])])
    visited = {start}
    m = controller.get_map()

    # Mark occupied tiles (other bots)
    occupied = set()
    for bid in controller.get_team_bot_ids():
        st = controller.get_bot_state(bid)
        if st:
            occupied.add((st['x'], st['y']))

    while queue:
        (x, y), path = queue.popleft()

        # Check if we're adjacent to goal
        if max(abs(x - gx), abs(y - gy)) <= 1:
            return path[0] if path else None

        # Explore 8 directions (Chebyshev)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < m.width and 0 <= ny < m.height):
                    continue
                if (nx, ny) in occupied and (nx, ny) != start:
                    continue
                if not m.tiles[nx][ny].is_walkable:
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))

    return None  # Unreachable


def find_tile(controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
    """Find first tile of given type."""
    m = controller.get_map()
    for x in range(m.width):
        for y in range(m.height):
            if m.tiles[x][y].tile_name == tile_name:
                return (x, y)
    return None


def find_empty_tile(controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
    """Find first empty tile of given type (no item on it)."""
    m = controller.get_map()
    for x in range(m.width):
        for y in range(m.height):
            tile = m.tiles[x][y]
            if tile.tile_name == tile_name and getattr(tile, "item", None) is None:
                return (x, y)
    return None


def find_item_on_tile(controller: RobotController, tile_name: str, item_check) -> Optional[Tuple[int, int]]:
    """Find tile with specific item. item_check is a lambda that takes item and returns bool."""
    m = controller.get_map()
    for x in range(m.width):
        for y in range(m.height):
            tile = m.tiles[x][y]
            if tile.tile_name == tile_name:
                item = getattr(tile, "item", None)
                if item and item_check(item):
                    return (x, y)
    return None


# ============================================================================
# COMMAND FRAMEWORK
# ============================================================================

class Command(ABC):
    """
    Base class for all bot commands.

    A command represents an atomic action that the bot should perform.
    Commands execute over multiple turns until completion.
    """

    @abstractmethod
    def execute(self, controller: RobotController, bot_id: int) -> None:
        """Execute one step of this command (called each turn)."""
        pass

    @abstractmethod
    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Check if command has completed successfully."""
        pass

    def on_start(self, controller: RobotController, bot_id: int) -> None:
        """Called when command becomes active (optional hook)."""
        pass

    def on_complete(self, controller: RobotController, bot_id: int) -> None:
        """Called when command completes (optional hook)."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable command description for debugging."""
        pass


# ============================================================================
# COMMAND IMPLEMENTATIONS
# ============================================================================

class BuyIngredientCommand(Command):
    """Command: Buy a specific ingredient from shop."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type
        self.shop_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']

        # Find shop
        if self.shop_loc is None:
            self.shop_loc = find_tile(controller, "SHOP")
            if not self.shop_loc:
                return

        sx, sy = self.shop_loc

        # Adjacent to shop? Buy it!
        if max(abs(bx - sx), abs(by - sy)) <= 1:
            controller.buy(bot_id, self.food_type, sx, sy)
            return

        # Navigate to shop
        next_move = bfs_to_adjacent(controller, (bx, by), self.shop_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        # holding is a dict with keys: type, food_name, food_id, chopped, cooked_stage
        return (holding is not None and
                holding.get('type') == 'Food' and
                holding.get('food_name') == self.food_type.name)

    def __str__(self) -> str:
        return f"BuyIngredient({self.food_type.name})"


class BuyPlateCommand(Command):
    """Command: Buy a plate from shop."""

    def __init__(self):
        self.shop_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']

        if self.shop_loc is None:
            self.shop_loc = find_tile(controller, "SHOP")
            if not self.shop_loc:
                return

        sx, sy = self.shop_loc

        if max(abs(bx - sx), abs(by - sy)) <= 1:
            controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.shop_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        # holding is a dict with keys: type, dirty, food (list)
        return holding is not None and holding.get('type') == 'Plate'

    def __str__(self) -> str:
        return "BuyPlate()"


class ChopIngredientCommand(Command):
    """Command: Chop held ingredient on counter."""

    def __init__(self):
        self.counter_loc = None
        self.state = "navigating"  # navigating → placing → chopping → picking_up → done

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if self.state == "navigating":
            if self.counter_loc is None:
                self.counter_loc = find_empty_tile(controller, "COUNTER")
                if not self.counter_loc:
                    return

            cx, cy = self.counter_loc
            if max(abs(bx - cx), abs(by - cy)) <= 1:
                self.state = "placing"
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.counter_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

        elif self.state == "placing":
            if holding is not None:
                cx, cy = self.counter_loc
                controller.place(bot_id, cx, cy)
                self.state = "chopping"

        elif self.state == "chopping":
            if controller.get_bot_state(bot_id).get('holding') is None:
                cx, cy = self.counter_loc
                controller.chop(bot_id, cx, cy)
                self.state = "picking_up"

        elif self.state == "picking_up":
            cx, cy = self.counter_loc
            controller.pickup(bot_id, cx, cy)
            self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        if self.state != "done":
            return False
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        # Check that we're holding chopped food (dict with chopped=True)
        return (holding is not None and
                holding.get('type') == 'Food' and
                holding.get('chopped') == True)

    def __str__(self) -> str:
        return "ChopIngredient()"


class CookIngredientCommand(Command):
    """Command: Cook held ingredient on cooker (place in pan, wait for cook, pickup)."""

    def __init__(self):
        self.cooker_loc = None
        self.state = "navigating"  # navigating → placing → cooking → picking_up → done

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if self.state == "navigating":
            if self.cooker_loc is None:
                self.cooker_loc = find_tile(controller, "COOKER")
                if not self.cooker_loc:
                    return

            cx, cy = self.cooker_loc
            if max(abs(bx - cx), abs(by - cy)) <= 1:
                self.state = "placing"
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.cooker_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

        elif self.state == "placing":
            if holding is not None:
                cx, cy = self.cooker_loc
                controller.place(bot_id, cx, cy)
                self.state = "cooking"

        elif self.state == "cooking":
            # Wait for cooking to complete (cooked_stage == 1)
            # Food is placed in a Pan on the cooker, so we need to check the Pan's food
            cx, cy = self.cooker_loc
            tile = controller.get_map().tiles[cx][cy]
            item = getattr(tile, "item", None)

            # Check if there's a pan with food cooking
            if item and item.__class__.__name__ == "Pan":
                food = getattr(item, "food", None)
                if food and hasattr(food, "cooked_stage") and food.cooked_stage == 1:
                    self.state = "picking_up"

        elif self.state == "picking_up":
            cx, cy = self.cooker_loc
            controller.take_from_pan(bot_id, cx, cy)
            self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        if self.state != "done":
            return False
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        # Check that we're holding cooked food (dict with cooked_stage=1)
        return (holding is not None and
                holding.get('type') == 'Food' and
                holding.get('cooked_stage') == 1)

    def __str__(self) -> str:
        return "CookIngredient()"


class StoreInBoxCommand(Command):
    """
    Command: Place held item in a box for storage.

    Note: Boxes can only store one type of item (e.g., all NOODLES or all MEAT).
    This command tries to find an appropriate box (empty or matching item type).
    """

    def __init__(self):
        self.box_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if holding is None:
            return

        if self.box_loc is None:
            # Find a suitable box: either empty or containing the same food type
            holding_food_name = holding.get('food_name') if holding and holding.get('type') == 'Food' else None

            m = controller.get_map()
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name == "BOX":
                        item = getattr(tile, "item", None)
                        # Accept if empty or if it contains the same food type
                        if item is None:
                            self.box_loc = (x, y)
                            break
                        elif hasattr(item, 'food_name') and item.food_name == holding_food_name:
                            self.box_loc = (x, y)
                            break
                if self.box_loc:
                    break

            if not self.box_loc:
                return

        box_x, box_y = self.box_loc

        if max(abs(bx - box_x), abs(by - box_y)) <= 1:
            controller.place(bot_id, box_x, box_y)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.box_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        return holding is None

    def __str__(self) -> str:
        return "StoreInBox()"


class PlaceOnCounterCommand(Command):
    """Command: Place held item on counter."""

    def __init__(self):
        self.counter_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if holding is None:
            return

        if self.counter_loc is None:
            self.counter_loc = find_empty_tile(controller, "COUNTER")
            if not self.counter_loc:
                return

        cx, cy = self.counter_loc

        if max(abs(bx - cx), abs(by - cy)) <= 1:
            controller.place(bot_id, cx, cy)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.counter_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        return bot_state.get('holding') is None

    def __str__(self) -> str:
        return "PlaceOnCounter()"


class PickupFromBoxCommand(Command):
    """Command: Pick up specific food type from box."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type
        self.box_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if holding is not None:
            return

        if self.box_loc is None:
            # Find box with our food
            self.box_loc = find_item_on_tile(
                controller,
                "BOX",
                lambda item: hasattr(item, 'food_name') and item.food_name == self.food_type.name
            )
            if not self.box_loc:
                return

        fx, fy = self.box_loc

        if max(abs(bx - fx), abs(by - fy)) <= 1:
            controller.pickup(bot_id, fx, fy)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.box_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        return holding and hasattr(holding, 'food_name') and holding.food_name == self.food_type.name

    def __str__(self) -> str:
        return f"PickupFromBox({self.food_type.name})"


class PickupPlateFromCounterCommand(Command):
    """Command: Pick up plate from counter."""

    def __init__(self):
        self.plate_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if holding is not None:
            return

        if self.plate_loc is None:
            # Find plate on counter (item is actual object, not dict)
            m = controller.get_map()
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name == "COUNTER":
                        item = getattr(tile, "item", None)
                        if item and item.__class__.__name__ == "Plate":
                            self.plate_loc = (x, y)
                            break
                if self.plate_loc:
                    break
            if not self.plate_loc:
                return

        px, py = self.plate_loc

        if max(abs(bx - px), abs(by - py)) <= 1:
            controller.pickup(bot_id, px, py)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.plate_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        # holding is a dict
        return holding is not None and holding.get('type') == 'Plate'

    def __str__(self) -> str:
        return "PickupPlateFromCounter()"


class AddFoodToPlateCommand(Command):
    """Command: Add food from counter to held plate."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type
        self.food_loc = None
        self.state = "navigating"

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # holding is a dict
        if holding is None or holding.get('type') != 'Plate':
            return

        if self.state == "navigating":
            if self.food_loc is None:
                # Find food on counter
                self.food_loc = find_item_on_tile(
                    controller,
                    "COUNTER",
                    lambda item: hasattr(item, 'food_name') and item.food_name == self.food_type.name
                )
                if not self.food_loc:
                    return

            fx, fy = self.food_loc
            if max(abs(bx - fx), abs(by - fy)) <= 1:
                self.state = "adding"
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.food_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

        elif self.state == "adding":
            fx, fy = self.food_loc
            controller.add_food_to_plate(bot_id, fx, fy)
            self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        return self.state == "done"

    def __str__(self) -> str:
        return f"AddFoodToPlate({self.food_type.name})"


class SubmitOrderCommand(Command):
    """Command: Submit completed plate at submit station."""

    def __init__(self, order_id: int = None):
        self.order_id = order_id  # Keep for reference, but not used in submit()
        self.submit_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if self.submit_loc is None:
            self.submit_loc = find_tile(controller, "SUBMIT")
            if not self.submit_loc:
                return

        sx, sy = self.submit_loc

        if max(abs(bx - sx), abs(by - sy)) <= 1:
            # holding is a dict
            if holding and holding.get('type') == 'Plate':
                # submit() automatically matches plate to orders
                controller.submit(bot_id, sx, sy)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.submit_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        # Submission complete when we're no longer holding the plate
        return bot_state.get('holding') is None

    def __str__(self) -> str:
        return f"SubmitOrder(order_id={self.order_id})"


# ============================================================================
# RECIPE PLANNER
# ============================================================================

class RecipePlanner:
    """
    Converts orders into command sequences.

    Strategy:
    1. Prepare all ingredients (buy → chop/cook as needed → store in box)
    2. Buy plate and place on counter
    3. Assembly loop: pickup ingredient → pickup plate → add to plate → place plate back
    4. Pickup final plate and submit
    """

    @staticmethod
    def needs_chopping(food_type: FoodType) -> bool:
        """Check if food type needs to be chopped."""
        return food_type.can_chop

    @staticmethod
    def needs_cooking(food_type: FoodType) -> bool:
        """Check if food type needs to be cooked."""
        return food_type.can_cook

    @staticmethod
    def build_commands_for_order(order: Dict) -> List[Command]:
        """
        Build command sequence to fulfill an order using plate-on-counter strategy.

        Strategy:
        1. Buy plate and place on counter
        2. For each ingredient:
           a. Buy ingredient
           b. Chop/cook as needed
           c. Place ingredient on counter
           d. Pick up plate
           e. Add ingredient to plate
           f. Place plate back on counter
        3. Pick up completed plate and submit

        Args:
            order: Dict with keys: order_id, required (list of food names)

        Returns:
            List of commands to execute in sequence
        """
        commands = []

        # Parse required foods
        required_foods = []
        for food_name in order["required"]:
            try:
                required_foods.append(FoodType[food_name])
            except KeyError:
                print(f"[WARN] Unknown food type: {food_name}")
                continue

        # Step 1: Buy plate and place on counter
        commands.append(BuyPlateCommand())
        commands.append(PlaceOnCounterCommand())

        # Step 2: For each ingredient, prepare and add to plate
        for food_type in required_foods:
            # Buy ingredient
            commands.append(BuyIngredientCommand(food_type))

            # Chop if needed
            if RecipePlanner.needs_chopping(food_type):
                commands.append(ChopIngredientCommand())

            # Cook if needed
            if RecipePlanner.needs_cooking(food_type):
                commands.append(CookIngredientCommand())

            # Place ingredient on counter (different from plate counter)
            commands.append(PlaceOnCounterCommand())

            # Pick up plate from counter
            commands.append(PickupPlateFromCounterCommand())

            # Add ingredient to plate (ingredient is on counter, plate is held)
            commands.append(AddFoodToPlateCommand(food_type))

            # Place plate back on counter
            commands.append(PlaceOnCounterCommand())

        # Step 3: Pick up completed plate and submit
        commands.append(PickupPlateFromCounterCommand())
        commands.append(SubmitOrderCommand(order["order_id"]))

        return commands


# ============================================================================
# BOT IMPLEMENTATION
# ============================================================================

class BotPlayer:
    """
    Command-based bot that uses a command queue to fulfill orders.
    """

    def __init__(self, map_copy):
        self.map = map_copy
        self.command_queue: List[Command] = []
        self.current_command_index = 0
        self.processed_orders = set()

    def get_current_command(self) -> Optional[Command]:
        """Get currently active command."""
        if self.current_command_index >= len(self.command_queue):
            return None
        return self.command_queue[self.current_command_index]

    def advance_command(self) -> None:
        """Move to next command."""
        self.current_command_index += 1

    def play_turn(self, controller: RobotController):
        """Main bot logic - called each turn."""
        bots = controller.get_team_bot_ids()
        if not bots:
            return

        bot_id = bots[0]  # Use first bot

        # If no commands, check for new orders
        if not self.command_queue or self.current_command_index >= len(self.command_queue):
            orders = controller.get_orders()

            # Find first active, unprocessed order
            for order in orders:
                if order["is_active"] and order["order_id"] not in self.processed_orders:
                    print(f"[BOT] Processing order {order['order_id']}: {order['required']}")
                    self.processed_orders.add(order["order_id"])

                    # Build command queue
                    self.command_queue = RecipePlanner.build_commands_for_order(order)
                    self.current_command_index = 0

                    print(f"[BOT] Generated {len(self.command_queue)} commands")
                    for i, cmd in enumerate(self.command_queue):
                        print(f"  {i}: {cmd}")
                    break

            if not self.command_queue:
                return

        # Execute current command
        current_cmd = self.get_current_command()
        if current_cmd is None:
            print("[BOT] All commands complete!")
            return

        # Execute
        current_cmd.execute(controller, bot_id)

        # Check completion
        if current_cmd.is_complete(controller, bot_id):
            print(f"[BOT] Command complete: {current_cmd}")
            current_cmd.on_complete(controller, bot_id)
            self.advance_command()
