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
    m = controller.get_map(controller.get_team())

    # Mark occupied tiles (only our own team bots to avoid collisions)
    # DON'T mark enemy bots as occupied - they might move!
    occupied = set()
    for bid in controller.get_team_bot_ids(controller.get_team()):
        st = controller.get_bot_state(bid)
        if st and (st['x'], st['y']) != start:  # Don't mark our own position as occupied
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
    m = controller.get_map(controller.get_team())
    for x in range(m.width):
        for y in range(m.height):
            if m.tiles[x][y].tile_name == tile_name:
                return (x, y)
    return None


def find_empty_tile(controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
    """Find first empty tile of given type (no item on it)."""
    m = controller.get_map(controller.get_team())
    for x in range(m.width):
        for y in range(m.height):
            tile = m.tiles[x][y]
            if tile.tile_name == tile_name and getattr(tile, "item", None) is None:
                return (x, y)
    return None


def find_item_on_tile(controller: RobotController, tile_name: str, item_check) -> Optional[Tuple[int, int]]:
    """Find tile with specific item. item_check is a lambda that takes item and returns bool."""
    m = controller.get_map(controller.get_team())
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
            # Wait for cooking to complete (cooked_stage == 1, perfectly cooked)
            # Food is placed in a Pan on the cooker, so we need to check the Pan's food
            cx, cy = self.cooker_loc
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
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
        # Check for properly cooked food (cooked_stage == 1)
        return (holding is not None and
                holding.get('type') == 'Food' and
                holding.get('cooked_stage') == 1)

    def __str__(self) -> str:
        return "CookIngredient()"


class StartCookingCommand(Command):
    """Command: Start cooking held ingredient (navigate to cooker and place in pan)."""

    def __init__(self):
        self.cooker_loc = None
        self.state = "navigating"  # navigating → placing → done

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
                self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        # Complete once we've placed food in cooker
        return self.state == "done"

    def __str__(self) -> str:
        return "StartCooking()"


class FinishCookingCommand(Command):
    """Command: Pick up cooked food from cooker when ready."""

    def __init__(self, food_type: FoodType):
        self.food_type = food_type
        self.cooker_loc = None
        self.state = "waiting"  # waiting → navigating → picking_up → done

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']

        if self.state == "waiting":
            # Find cooker with our food
            if self.cooker_loc is None:
                self.cooker_loc = self._find_cooker_with_food(controller)
                if not self.cooker_loc:
                    return

            # Check if food is cooked (stage == 1: perfectly cooked, not burnt)
            cx, cy = self.cooker_loc
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            item = getattr(tile, "item", None)

            if item and item.__class__.__name__ == "Pan":
                food = getattr(item, "food", None)
                # Only pick up when cooked_stage == 1 (perfectly cooked)
                # If >= 2, the food burned while we were doing other tasks
                if food and hasattr(food, "cooked_stage") and food.cooked_stage == 1:
                    self.state = "navigating"

        elif self.state == "navigating":
            cx, cy = self.cooker_loc
            if max(abs(bx - cx), abs(by - cy)) <= 1:
                self.state = "picking_up"
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.cooker_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

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
        # Check for properly cooked food (cooked_stage == 1)
        return (holding is not None and
                holding.get('type') == 'Food' and
                holding.get('cooked_stage') == 1)

    def _find_cooker_with_food(self, controller: RobotController) -> Optional[Tuple[int, int]]:
        """Find cooker with our food type cooking."""
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COOKER":
                    item = getattr(tile, "item", None)
                    if item and item.__class__.__name__ == "Pan":
                        food = getattr(item, "food", None)
                        if food and hasattr(food, 'food_name') and food.food_name == self.food_type.name:
                            return (x, y)
        return None

    def __str__(self) -> str:
        return f"FinishCooking({self.food_type.name})"


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

            m = controller.get_map(controller.get_team())
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
            m = controller.get_map(controller.get_team())
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


class PlaceInBoxCommand(Command):
    """Command: Place held item (plate or food) in a box."""

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
            # Find a suitable box: either empty or containing the same item type
            holding_type = holding.get('type') if isinstance(holding, dict) else holding.__class__.__name__

            m = controller.get_map(controller.get_team())
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name == "BOX":
                        item = getattr(tile, "item", None)
                        # Accept if empty
                        if item is None:
                            self.box_loc = (x, y)
                            break
                        # Accept if it contains the same type (Plate or specific food)
                        elif item.__class__.__name__ == holding_type:
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
        return "PlaceInBox()"


class PickupPlateFromBoxCommand(Command):
    """Command: Pick up plate from box."""

    def __init__(self):
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
            # Find box with plate
            m = controller.get_map(controller.get_team())
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name == "BOX":
                        item = getattr(tile, "item", None)
                        if item and item.__class__.__name__ == "Plate":
                            self.box_loc = (x, y)
                            break
                if self.box_loc:
                    break
            if not self.box_loc:
                return

        px, py = self.box_loc

        if max(abs(bx - px), abs(by - py)) <= 1:
            controller.pickup(bot_id, px, py)
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.box_loc)
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
        return "PickupPlateFromBox()"


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
        self.failed_attempts = 0  # Track failed submit attempts

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
                # Check if we've been stuck trying to submit for too long
                if self.failed_attempts >= 3:
                    print(f"[SUBMIT ERROR] Failed to submit {self.failed_attempts} times - plate doesn't match any order!")
                    # Trash the plate by placing it anywhere (it will be cleaned up later)
                    # Don't call submit again
                    return

                # submit() automatically matches plate to orders
                success = controller.submit(bot_id, sx, sy)
                if not success:
                    self.failed_attempts += 1
                    print(f"[SUBMIT] Submit failed (attempt {self.failed_attempts}/3)")
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.submit_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        # Submission complete when we're no longer holding the plate
        # OR if we've failed too many times (give up)
        if self.failed_attempts >= 3:
            return True
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
    def count_tiles(controller: RobotController, tile_name: str) -> int:
        """Count number of tiles of a given type on the map."""
        m = controller.get_map(controller.get_team())
        count = 0
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].tile_name == tile_name:
                    count += 1
        return count

    @staticmethod
    def build_commands_for_order(order: Dict, controller: RobotController = None) -> List[Command]:
        """
        Build command sequence to fulfill an order with optimized cooking strategy.

        Strategy (optimized for cooking efficiency):
        1. Buy plate and place on counter (or box if only 1 counter)
        2. Separate ingredients into cooking and non-cooking groups
        3. For cooking ingredients:
           - Buy, chop (if needed), start cooking
           - While cooking, prepare non-cooking ingredients
           - Complete cooking and add to plate
        4. Add remaining non-cooking ingredients to plate
        5. Pick up completed plate and submit

        Args:
            order: Dict with keys: order_id, required (list of food names)
            controller: RobotController instance (optional, for tile counting)

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

        # Separate into cooking and non-cooking ingredients
        cooking_foods = [f for f in required_foods if RecipePlanner.needs_cooking(f)]
        non_cooking_foods = [f for f in required_foods if not RecipePlanner.needs_cooking(f)]

        # Detect if we have limited counters (1 counter scenario)
        use_box_for_plate = False
        plate_already_exists = False
        if controller is not None:
            counter_count = RecipePlanner.count_tiles(controller, "COUNTER")
            box_count = RecipePlanner.count_tiles(controller, "BOX")
            # If only 1 counter and at least 1 box, use box for plate storage
            if counter_count == 1 and box_count >= 1:
                use_box_for_plate = True
                print(f"[BOT] Detected 1 counter + {box_count} box(es) - using box for plate storage")

            # Check if a clean, empty plate already exists on counter or in box
            m = controller.get_map(controller.get_team())
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name in ["COUNTER", "BOX"]:
                        item = getattr(tile, "item", None)
                        if item and item.__class__.__name__ == "Plate":
                            # Check if plate is clean and empty
                            is_dirty = getattr(item, "dirty", False)
                            food_on_plate = getattr(item, "food", [])
                            if not is_dirty and len(food_on_plate) == 0:
                                plate_already_exists = True
                                # Determine where the plate is stored
                                if tile.tile_name == "BOX":
                                    use_box_for_plate = True
                                print(f"[BOT] Found existing clean empty plate on {tile.tile_name} at ({x},{y}) - skipping plate purchase")
                                break
                if plate_already_exists:
                    break

        # Step 1: Buy plate and place in appropriate location (only if no clean empty plate exists)
        if not plate_already_exists:
            commands.append(BuyPlateCommand())
            if use_box_for_plate:
                commands.append(PlaceInBoxCommand())
            else:
                commands.append(PlaceOnCounterCommand())

        # Helper function to add ingredient to plate (varies based on plate storage location)
        def add_ingredient_to_plate_sequence(food_type: FoodType) -> List[Command]:
            """Generate command sequence to add an ingredient to the plate."""
            seq = []
            # Place ingredient on counter
            seq.append(PlaceOnCounterCommand())

            # Pick up plate from its storage location
            if use_box_for_plate:
                seq.append(PickupPlateFromBoxCommand())
            else:
                seq.append(PickupPlateFromCounterCommand())

            # Add ingredient to plate
            seq.append(AddFoodToPlateCommand(food_type))

            # Place plate back in storage
            if use_box_for_plate:
                seq.append(PlaceInBoxCommand())
            else:
                seq.append(PlaceOnCounterCommand())

            return seq

        # Step 2: Process ingredients
        # If using box strategy (1 counter), don't parallelize cooking to avoid burning
        if cooking_foods and not use_box_for_plate:
            # PARALLELIZED COOKING STRATEGY (multiple counters available)
            # Start with first cooking ingredient
            first_cooking = cooking_foods[0]

            # Buy and prep first cooking ingredient
            commands.append(BuyIngredientCommand(first_cooking))
            if RecipePlanner.needs_chopping(first_cooking):
                commands.append(ChopIngredientCommand())

            # START cooking (place in cooker but don't wait)
            commands.append(StartCookingCommand())

            # While food is cooking, prepare all non-cooking ingredients and add to plate
            for food_type in non_cooking_foods:
                # Buy ingredient
                commands.append(BuyIngredientCommand(food_type))

                # Chop if needed
                if RecipePlanner.needs_chopping(food_type):
                    commands.append(ChopIngredientCommand())

                # Add ingredient to plate (sequence varies based on plate storage)
                commands.extend(add_ingredient_to_plate_sequence(food_type))

            # FINISH cooking - pick up when ready (will wait if not done yet)
            commands.append(FinishCookingCommand(first_cooking))

            # Add the cooked ingredient to plate
            commands.extend(add_ingredient_to_plate_sequence(first_cooking))

            # Process remaining cooking ingredients (if any)
            for food_type in cooking_foods[1:]:
                commands.append(BuyIngredientCommand(food_type))

                if RecipePlanner.needs_chopping(food_type):
                    commands.append(ChopIngredientCommand())

                commands.append(CookIngredientCommand())

                # Add to plate
                commands.extend(add_ingredient_to_plate_sequence(food_type))
        else:
            # SEQUENTIAL STRATEGY (1 counter or no cooking ingredients)
            # Process all ingredients sequentially without parallelizing
            for food_type in required_foods:
                # Buy ingredient
                commands.append(BuyIngredientCommand(food_type))

                # Chop if needed
                if RecipePlanner.needs_chopping(food_type):
                    commands.append(ChopIngredientCommand())

                # Cook if needed (wait for completion before continuing)
                if RecipePlanner.needs_cooking(food_type):
                    commands.append(CookIngredientCommand())

                # Add to plate
                commands.extend(add_ingredient_to_plate_sequence(food_type))

        # Step 3: Pick up completed plate and submit
        if use_box_for_plate:
            commands.append(PickupPlateFromBoxCommand())
        else:
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
        self.current_order_id = None  # Track which order we're currently working on
        self.cleaning_workspace = False  # Track if we're cleaning up after expired order
        self.sabotage_mode = False  # Track if we're in enemy kitchen sabotaging
        self.has_switched = False  # Track if we've already switched once
        self.sabotage_target_turn = 250  # Turn to switch when window opens
        self.returned_from_sabotage = False  # Track if we've returned and need cleanup
        self.plates_placed = 0  # Track how many plates we've placed for sabotage
        self.sabotage_stuck_counter = 0  # Track how many turns we've been stuck
        self.sabotage_skip_targets = set()  # Set of targets to skip if we're stuck on them
        self.sabotage_last_target = None  # Last target we tried to reach
        self.sabotage_recursion_depth = 0  # Track recursion depth to prevent infinite loops

    def get_current_command(self) -> Optional[Command]:
        """Get currently active command."""
        if self.current_command_index >= len(self.command_queue):
            return None
        return self.command_queue[self.current_command_index]

    def advance_command(self) -> None:
        """Move to next command."""
        self.current_command_index += 1

    def _execute_sabotage(self, controller: RobotController, bot_id: int) -> None:
        """Execute sabotage strategy: steal pans and items from counters, trash them or hide in enemy boxes."""
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')
        current_turn = controller.get_turn()

        # Debug: log position and holding status every 10 turns
        if current_turn % 10 == 0:
            holding_str = holding.get('type') if holding else "nothing"
            print(f"[SABOTAGE DEBUG] Turn {current_turn}: Bot at ({bx},{by}), holding {holding_str}")

        # If holding something, try to put pans in enemy boxes, otherwise trash
        if holding is not None:
            holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

            # Special handling for Pans - try to hide them in enemy's box first
            if holding_type == 'Pan':
                box_loc = find_empty_tile(controller, "BOX")
                if not box_loc:
                    box_loc = find_tile(controller, "BOX")

                if box_loc:
                    bx_loc, by_loc = box_loc
                    if max(abs(bx - bx_loc), abs(by - by_loc)) <= 1:
                        success = controller.place(bot_id, bx_loc, by_loc)
                        if success:
                            print(f"[SABOTAGE] Hid Pan in enemy's box at ({bx_loc},{by_loc})!")
                        return
                    else:
                        # Navigate to box
                        next_move = bfs_to_adjacent(controller, (bx, by), box_loc)
                        if next_move:
                            controller.move(bot_id, next_move[0], next_move[1])
                        return
                # If no box available, we'll take it back with us (fall through to end of sabotage)
                print(f"[SABOTAGE] No enemy box available for Pan, will take it back")
                return

            # For non-Pan items, trash them
            trash_loc = find_tile(controller, "TRASH")
            if not trash_loc:
                print(f"[SABOTAGE] ERROR: No trash found in enemy map!")
                return

            tx, ty = trash_loc
            if max(abs(bx - tx), abs(by - ty)) <= 1:
                # Adjacent to trash, throw away item
                success = controller.trash(bot_id, tx, ty)
                if success:
                    holding_type = holding.get('type') if isinstance(holding, dict) else holding.__class__.__name__
                    print(f"[SABOTAGE] Trashed enemy's {holding_type}!")
                return
            else:
                # Navigate to trash
                next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])
                return

        # Not holding anything - find something to steal
        # Priority: 1. Pans with food (cooking in progress), 2. Food/plates on counters
        m = controller.get_map(controller.get_team())

        # Search for items to steal
        targets = []  # List of (priority, distance, x, y, description)

        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                item = getattr(tile, "item", None)

                if item is None:
                    continue

                item_class = item.__class__.__name__
                distance = max(abs(bx - x), abs(by - y))

                # Priority 1: Pans (with or without food) on COOKER
                if tile.tile_name == "COOKER" and item_class == "Pan":
                    food = getattr(item, "food", None)
                    if food:
                        targets.append((10, -distance, x, y, f"Pan with {food.food_name} on COOKER"))
                    else:
                        targets.append((8, -distance, x, y, "Empty Pan on COOKER"))

                # Priority 2: Food on COUNTER (prepared ingredients)
                elif tile.tile_name == "COUNTER" and item_class == "Food":
                    food_name = getattr(item, "food_name", "Unknown")
                    targets.append((7, -distance, x, y, f"{food_name} on COUNTER"))

                # Priority 3: Plates with food on COUNTER
                elif tile.tile_name == "COUNTER" and item_class == "Plate":
                    foods = getattr(item, "food", [])
                    if foods:
                        targets.append((6, -distance, x, y, f"Plate with {len(foods)} items on COUNTER"))
                    else:
                        targets.append((5, -distance, x, y, "Empty plate on COUNTER"))

        # Sort by priority (higher first), then distance (closer first)
        targets.sort(reverse=True)

        if not targets:
            print(f"[SABOTAGE] No items found to steal! Enemy kitchen is clean.")
            return

        # Try to steal the highest priority target
        for priority, neg_dist, tx, ty, description in targets:
            if max(abs(bx - tx), abs(by - ty)) <= 1:
                # Adjacent - try to pick up
                success = controller.pickup(bot_id, tx, ty)
                if success:
                    print(f"[SABOTAGE] Stole: {description}")
                return
            else:
                # Navigate toward target
                next_move = bfs_to_adjacent(controller, (bx, by), (tx, ty))
                if next_move:
                    success = controller.move(bot_id, next_move[0], next_move[1])
                    if success:
                        print(f"[SABOTAGE] Moving to steal: {description}")
                    return

        # Couldn't reach any targets
        print(f"[SABOTAGE] Found {len(targets)} items but couldn't reach any")

    def play_turn(self, controller: RobotController):
        """Main bot logic - called each turn."""
        bots = controller.get_team_bot_ids(controller.get_team())
        if not bots:
            return

        current_turn = controller.get_turn()
        switch_info = controller.get_switch_info()

        bot_id = bots[0]

        # SABOTAGE LOGIC: Always use Bot 0 for switching to enemy map
        # IMPORTANT: Only switch when bot 0 isn't holding anything
        bot0_state = controller.get_bot_state(bots[0])
        bot0_holding = bot0_state.get('holding') if bot0_state else None

        if not self.has_switched and current_turn >= self.sabotage_target_turn and controller.can_switch_maps():
            # Only switch if Bot 0's hands are empty
            if bot0_holding is None:
                print(f"[SABOTAGE] Turn {current_turn}: Bot 0 switching to enemy map for sabotage!")
                if controller.switch_maps():
                    self.has_switched = True
                    self.sabotage_mode = True
                    # DON'T clear command queue - we'll resume it when we return
                    # Just note that we're in sabotage mode now
                    print(f"[SABOTAGE] Successfully switched! Will sabotage until turn {switch_info['window_end_turn']}")
                    print(f"[SABOTAGE] Saved command queue state: {len(self.command_queue)} commands, index {self.current_command_index}, order {self.current_order_id}")
                else:
                    print(f"[SABOTAGE] Failed to switch maps!")
            else:
                if current_turn % 10 == 0:  # Log every 10 turns to avoid spam
                    print(f"[SABOTAGE] Turn {current_turn}: Waiting to switch - Bot 0 is holding {bot0_holding.get('type')}")

        # Check if we just returned from sabotage
        if self.sabotage_mode and not switch_info['window_active']:
            print(f"[SABOTAGE] Turn {current_turn}: Returned to our kitchen")
            self.sabotage_mode = False
            self.returned_from_sabotage = True

            # If Bot 0 is holding anything (stolen from enemy), handle it
            bot0_state = controller.get_bot_state(bots[0])
            if bot0_state and bot0_state.get('holding'):
                holding = bot0_state.get('holding')
                holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

                # If holding a Pan, we can't trash it - place it in an empty box instead
                if holding_type == 'Pan':
                    print(f"[SABOTAGE] Returned with Pan - placing in box")
                    box_loc = find_empty_tile(controller, "BOX")
                    if not box_loc:
                        # No empty box, try any box
                        box_loc = find_tile(controller, "BOX")
                    if box_loc:
                        bx, by = bot0_state['x'], bot0_state['y']
                        bx_loc, by_loc = box_loc
                        if max(abs(bx - bx_loc), abs(by - by_loc)) <= 1:
                            controller.place(bots[0], bx_loc, by_loc)
                            print(f"[SABOTAGE] Placed Pan in box at ({bx_loc},{by_loc})")
                            # Successfully placed pan, clear the flag and fall through to normal operation
                        else:
                            # Navigate to box
                            print(f"[SABOTAGE] Navigating to box to place Pan")
                            next_move = bfs_to_adjacent(controller, (bx, by), box_loc)
                            if next_move:
                                controller.move(bots[0], next_move[0], next_move[1])
                            return
                else:
                    # For other items (Food, Plate), try to trash them
                    trash_loc = find_tile(controller, "TRASH")
                    if trash_loc:
                        bx, by = bot0_state['x'], bot0_state['y']
                        tx, ty = trash_loc
                        if max(abs(bx - tx), abs(by - ty)) <= 1:
                            controller.trash(bots[0], tx, ty)
                            print(f"[SABOTAGE] Trashed held {holding_type} on return")
                            # Successfully trashed item, clear the flag and fall through to normal operation
                        else:
                            # Not adjacent to trash, navigate there
                            print(f"[SABOTAGE] Navigating to trash {holding_type}")
                            next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                            if next_move:
                                controller.move(bots[0], next_move[0], next_move[1])
                            return

            # Check if the order we were working on expired during sabotage
            if self.current_order_id is not None:
                orders = controller.get_orders(controller.get_team())
                order_still_active = any(o["order_id"] == self.current_order_id and o["is_active"] for o in orders)

                if not order_still_active:
                    print(f"[SABOTAGE] Order {self.current_order_id} expired during sabotage - clearing workspace")
                    # Clear the command queue and enter cleanup mode
                    self.command_queue = []
                    self.current_command_index = 0
                    self.current_order_id = None
                    self.cleaning_workspace = True
                    self.returned_from_sabotage = False
                    # Fall through to cleanup logic below
                else:
                    print(f"[SABOTAGE] Order {self.current_order_id} still active - but commands have stale state after teleport")
                    # The command queue has internal state (positions, etc.) that's now invalid after teleporting
                    # We need to regenerate the command queue for the same order
                    order_obj = next((o for o in orders if o["order_id"] == self.current_order_id), None)
                    if order_obj:
                        print(f"[SABOTAGE] Regenerating command queue for order {self.current_order_id}")
                        self.command_queue = RecipePlanner.build_commands_for_order(order_obj, controller)
                        self.current_command_index = 0
                        print(f"[SABOTAGE] Regenerated {len(self.command_queue)} commands")
                    else:
                        print(f"[SABOTAGE] ERROR: Could not find order {self.current_order_id}!")
                        self.command_queue = []
                        self.current_command_index = 0
                        self.current_order_id = None
                    self.returned_from_sabotage = False
            else:
                # No order was active, just reset the flag
                self.returned_from_sabotage = False

            print(f"[SABOTAGE] Bot 0 will resume normal operations")
            # Don't return - fall through to normal order processing

        # If in sabotage mode, execute sabotage strategy with Bot 0
        if self.sabotage_mode:
            self.sabotage_recursion_depth = 0  # Reset recursion depth at start of turn
            self._execute_sabotage(controller, bots[0])
            return

        # Check if current order is still active (hasn't expired)
        if self.current_order_id is not None:
            orders = controller.get_orders(controller.get_team())
            order_still_active = any(o["order_id"] == self.current_order_id and o["is_active"] for o in orders)

            if not order_still_active:
                print(f"[BOT] Turn {controller.get_turn()}: Order {self.current_order_id} expired, clearing workspace")

                # Clear command queue and enter cleanup mode
                self.command_queue = []
                self.current_command_index = 0
                self.current_order_id = None
                self.cleaning_workspace = True  # Enter cleanup mode
                # Cleanup logic will run on next section

        # Check if we're in cleanup mode (clearing workspace after expired order or sabotage return)
        if self.cleaning_workspace:
            # Continue cleanup process
            bot_state = controller.get_bot_state(bot_id)
            if not bot_state:
                return

            bx, by = bot_state['x'], bot_state['y']
            holding = bot_state.get('holding')

            # If holding something, get rid of it first
            if holding is not None:
                holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

                # Trash everything we're holding during cleanup
                trash_loc = find_tile(controller, "TRASH")
                if trash_loc:
                    tx, ty = trash_loc
                    if max(abs(bx - tx), abs(by - ty)) <= 1:
                        controller.trash(bot_id, tx, ty)
                        print(f"[CLEANUP] Trashed {holding_type}")
                        return
                    else:
                        next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                        if next_move:
                            controller.move(bot_id, next_move[0], next_move[1])
                        return
                else:
                    # No trash tile! Fallback: place on any empty counter/box
                    print(f"[CLEANUP] WARNING: No trash found, placing {holding_type} on counter/box")
                    m = controller.get_map(controller.get_team())
                    for tile_name in ["COUNTER", "BOX"]:
                        for x in range(m.width):
                            for y in range(m.height):
                                tile = m.tiles[x][y]
                                if tile.tile_name == tile_name and getattr(tile, "item", None) is None:
                                    if max(abs(bx - x), abs(by - y)) <= 1:
                                        controller.place(bot_id, x, y)
                                        print(f"[CLEANUP] Placed {holding_type} on {tile_name}")
                                        return
                                    else:
                                        next_move = bfs_to_adjacent(controller, (bx, by), (x, y))
                                        if next_move:
                                            controller.move(bot_id, next_move[0], next_move[1])
                                        return
                    # Can't find anywhere to place - just stay holding and try next turn
                    print(f"[CLEANUP] ERROR: Can't find anywhere to place {holding_type}!")
                    return

            # Hands are empty, clean up items from critical surfaces
            # Priority: SUBMIT > COUNTER > BOX
            # NOTE: Do NOT clean COOKER (pans supposed to be there), SINK (dirty plates), or SINKTABLE (generates clean plates!)
            m = controller.get_map(controller.get_team())
            cleanup_target = None
            cleanup_surfaces = ["SUBMIT", "COUNTER", "BOX"]

            # Count how many plates we need to clean
            plates_found = []
            for surface_type in cleanup_surfaces:
                for x in range(m.width):
                    for y in range(m.height):
                        tile = m.tiles[x][y]
                        if tile.tile_name == surface_type:
                            item = getattr(tile, "item", None)
                            # Only pick up Plates (enemy sabotage) - don't pick up food/pans
                            # CRITICAL: SINKTABLE has infinite clean plates, so checking tile_name is important!
                            if item is not None and hasattr(item, '__class__') and item.__class__.__name__ == 'Plate':
                                plates_found.append((x, y, tile.tile_name))
                                if cleanup_target is None:
                                    cleanup_target = (x, y)

            if plates_found:
                print(f"[CLEANUP DEBUG] Found {len(plates_found)} plates to clean at: {plates_found}")

            if cleanup_target:
                cx, cy = cleanup_target
                if max(abs(bx - cx), abs(by - cy)) <= 1:
                    # Only pickup if hands are empty (otherwise we're in an infinite loop!)
                    if holding is None:
                        tile_name_before = m.tiles[cx][cy].tile_name
                        item_before = m.tiles[cx][cy].item
                        item_type_before = item_before.__class__.__name__ if item_before else "Unknown"
                        pickup_success = controller.pickup(bot_id, cx, cy)
                        print(f"[CLEANUP] Picked up {item_type_before} from {tile_name_before} at ({cx},{cy}), success={pickup_success}")
                    else:
                        # Already holding something - this shouldn't happen, but handle it
                        print(f"[CLEANUP] ERROR: Tried to pickup but already holding {holding.get('type')}!")
                    return
                else:
                    next_move = bfs_to_adjacent(controller, (bx, by), cleanup_target)
                    if next_move:
                        controller.move(bot_id, next_move[0], next_move[1])
                    return

            # All critical surfaces are clean
            print(f"[CLEANUP] All critical surfaces cleared, ready for new orders")
            self.cleaning_workspace = False
            self.returned_from_sabotage = False  # Reset flag
            # DON'T return - fall through to order processing below!

        # If no commands, check for new orders
        # BUT: only if bot has empty hands (don't start new orders while holding items from expired orders)
        if not self.command_queue or self.current_command_index >= len(self.command_queue):
            bot_state = controller.get_bot_state(bot_id)
            holding_check = bot_state.get('holding') if bot_state else None
            print(f"[BOT DEBUG] Command queue empty check: holding={holding_check}")
            if bot_state and bot_state.get('holding') is not None:
                # Still holding something from a previous order, need to get rid of it
                holding = bot_state.get('holding')
                holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

                # If holding a Plate or Pan, try to place it somewhere (can't trash these)
                if holding_type == 'Plate' or holding_type == 'Pan':
                    print(f"[BOT] Holding {holding_type}, will place in box to clear hands")
                    # Find an empty box to dump the item
                    target_loc = find_empty_tile(controller, "BOX")
                    if not target_loc:
                        # No empty box, try any box (can stack)
                        target_loc = find_tile(controller, "BOX")
                    if not target_loc:
                        # No box, try counter as fallback
                        target_loc = find_empty_tile(controller, "COUNTER")

                    if target_loc:
                        cx, cy = target_loc
                        bx, by = bot_state['x'], bot_state['y']

                        if max(abs(bx - cx), abs(by - cy)) <= 1:
                            controller.place(bot_id, cx, cy)
                            print(f"[BOT] Placed {holding_type}, hands now free")
                        else:
                            next_move = bfs_to_adjacent(controller, (bx, by), target_loc)
                            if next_move:
                                controller.move(bot_id, next_move[0], next_move[1])
                else:
                    # For other items (Food), trash them
                    print(f"[BOT] Holding {holding_type}, navigating to trash")
                    trash_loc = find_tile(controller, "TRASH")
                    if trash_loc:
                        tx, ty = trash_loc
                        bx, by = bot_state['x'], bot_state['y']

                        if max(abs(bx - tx), abs(by - ty)) <= 1:
                            trash_result = controller.trash(bot_id, tx, ty)
                            print(f"[BOT] Trashed {holding_type}, result={trash_result}")
                            # Verify hands are now empty
                            bot_state_after = controller.get_bot_state(bot_id)
                            holding_after = bot_state_after.get('holding') if bot_state_after else None
                            print(f"[BOT DEBUG] After trash: holding={holding_after}")
                        else:
                            next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                            if next_move:
                                controller.move(bot_id, next_move[0], next_move[1])
                return

            orders = controller.get_orders(controller.get_team())
            current_turn = controller.get_turn()

            # Collect all valid orders with their metrics
            candidate_orders = []

            for order in orders:
                if order["is_active"] and order["order_id"] not in self.processed_orders:
                    # Calculate remaining time for this order
                    expires_turn = order.get("expires_turn", float('inf'))
                    remaining_turns = expires_turn - current_turn

                    # Check if order has cooking ingredients
                    has_cooking = any(
                        RecipePlanner.needs_cooking(FoodType[food_name])
                        for food_name in order["required"]
                        if food_name in [ft.name for ft in FoodType]
                    )

                    # Skip orders without enough time
                    min_turns_needed = 50 if has_cooking else 10
                    if remaining_turns < min_turns_needed:
                        print(f"[BOT] Skipping order {order['order_id']} - only {remaining_turns} turns remaining (need {min_turns_needed})")
                        self.processed_orders.add(order["order_id"])  # Mark as processed so we don't try again
                        continue

                    # Complexity check: estimate turns needed based on ingredient count
                    num_ingredients = len(order["required"])
                    cooking_count = sum(1 for food_name in order["required"]
                                       if food_name in [ft.name for ft in FoodType]
                                       and RecipePlanner.needs_cooking(FoodType[food_name]))

                    # Empirical formula: ~35 turns per ingredient + 30 per cooking ingredient
                    estimated_turns = num_ingredients * 35 + cooking_count * 30

                    # Skip orders where we likely don't have enough time (need 1.2x safety margin)
                    if remaining_turns < estimated_turns * 1.2:
                        print(f"[BOT] Skipping order {order['order_id']} - too complex ({num_ingredients} ingredients, {cooking_count} cooking) "
                              f"for {remaining_turns} turns (need ~{int(estimated_turns * 1.2)})")
                        self.processed_orders.add(order["order_id"])
                        continue

                    # Calculate value (reward + penalty)
                    reward = order.get('reward', 0)
                    penalty = order.get('penalty', 0)
                    total_value = reward + penalty

                    candidate_orders.append({
                        'order': order,
                        'expires_turn': expires_turn,
                        'remaining_turns': remaining_turns,
                        'total_value': total_value,
                        'reward': reward,
                        'penalty': penalty
                    })

            if not candidate_orders:
                return

            # Strategy: Find the earliest expiring order, then consider all orders
            # that expire within 50 turns of it, and pick the highest value one
            earliest_expiry = min(c['expires_turn'] for c in candidate_orders)
            deadline_window = earliest_expiry + 50

            # Filter to orders within the deadline window
            orders_in_window = [c for c in candidate_orders if c['expires_turn'] <= deadline_window]

            # Pick the highest value order in the window
            best = max(orders_in_window, key=lambda c: c['total_value'])

            order = best['order']
            print(f"[BOT] Turn {current_turn}: Processing order {order['order_id']}: {order['required']} "
                  f"({best['remaining_turns']} turns remaining, expires at turn {best['expires_turn']}, "
                  f"value=${best['total_value']})")

            self.processed_orders.add(order["order_id"])
            self.current_order_id = order["order_id"]

            # Build command queue (pass controller to detect tile availability)
            self.command_queue = RecipePlanner.build_commands_for_order(order, controller)
            self.current_command_index = 0

            print(f"[BOT] Generated {len(self.command_queue)} commands")
            for i, cmd in enumerate(self.command_queue):
                print(f"  {i}: {cmd}")

            if not self.command_queue:
                return

        # Execute current command
        current_cmd = self.get_current_command()
        if current_cmd is None:
            print("[BOT] All commands complete!")
            self.current_order_id = None  # Clear current order when all commands done
            return

        # Execute
        current_cmd.execute(controller, bot_id)

        # Check completion
        if current_cmd.is_complete(controller, bot_id):
            print(f"[BOT] Command complete: {current_cmd}")
            current_cmd.on_complete(controller, bot_id)
            self.advance_command()

            # If we just finished the last command, clear current order
            if self.current_command_index >= len(self.command_queue):
                self.current_order_id = None
