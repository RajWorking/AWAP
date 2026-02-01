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

    # Mark occupied tiles (other bots)
    occupied = set()
    for bid in controller.get_team_bot_ids(controller.get_team()):
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


def find_closest_tile(controller: RobotController, tile_name: str, from_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Find closest tile of given type to a position."""
    m = controller.get_map(controller.get_team())
    closest_tile = None
    min_dist = float('inf')

    fx, fy = from_pos
    for x in range(m.width):
        for y in range(m.height):
            if m.tiles[x][y].tile_name == tile_name:
                # Use Chebyshev distance (max of abs differences)
                dist = max(abs(x - fx), abs(y - fy))
                if dist < min_dist:
                    min_dist = dist
                    closest_tile = (x, y)

    return closest_tile


def find_closest_empty_tile(controller: RobotController, tile_name: str, from_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Find closest empty tile of given type to a position."""
    m = controller.get_map(controller.get_team())
    closest_tile = None
    min_dist = float('inf')

    fx, fy = from_pos
    for x in range(m.width):
        for y in range(m.height):
            tile = m.tiles[x][y]
            if tile.tile_name == tile_name and getattr(tile, "item", None) is None:
                # Use Chebyshev distance (max of abs differences)
                dist = max(abs(x - fx), abs(y - fy))
                if dist < min_dist:
                    min_dist = dist
                    closest_tile = (x, y)

    return closest_tile


# ============================================================================
# COMMAND FRAMEWORK
# ============================================================================

class Command(ABC):
    """
    Base class for all bot commands.

    A command represents an atomic action that the bot should perform.
    Commands execute over multiple turns until completion.
    """

    def __init__(self):
        self.turn_count = 0  # Track how many turns this command has been executing
        self.max_turns = 100  # Maximum turns before declaring command stuck
        self.last_state = None  # Track last state for progress detection
        self.stuck_counter = 0  # Track consecutive turns without progress

    @abstractmethod
    def execute(self, controller: RobotController, bot_id: int) -> None:
        """Execute one step of this command (called each turn)."""
        pass

    @abstractmethod
    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Check if command has completed successfully."""
        pass

    def is_stuck(self) -> bool:
        """Check if command has been executing for too long."""
        # Check if exceeded max turns
        if self.turn_count > self.max_turns:
            return True
        # Check if truly stuck (no progress for extended time)
        if self.stuck_counter > 20:  # Increased threshold
            return True
        # Check if failed explicitly
        if hasattr(self, 'failed') and self.failed:
            return True
        return False

    def check_progress(self, current_state: str) -> None:
        """Track progress to detect if command is stuck in same state."""
        if current_state == self.last_state:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_state = current_state

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
        super().__init__()
        self.food_type = food_type
        self.shop_loc = None
        self.max_turns = 30  # Reduce timeout for buying
        self.buy_attempted = False

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress
        self.check_progress(f"{bx}_{by}_{holding is not None}")

        # Can't buy if already holding something
        if holding is not None:
            if self.turn_count > 3:  # Give a few turns grace period
                print(f"[WARN] BuyIngredient: Still holding item, can't buy")
            return

        # Find shop
        if self.shop_loc is None:
            self.shop_loc = find_tile(controller, "SHOP")
            if not self.shop_loc:
                print(f"[WARN] BuyIngredient: No SHOP tile found")
                return

        sx, sy = self.shop_loc

        # Adjacent to shop? Buy it!
        if max(abs(bx - sx), abs(by - sy)) <= 1:
            if not self.buy_attempted:
                # Check if we can afford it
                current_money = controller.get_team_money(controller.get_team())
                if current_money < self.food_type.buy_cost:
                    print(f"[WARN] BuyIngredient: Not enough money (have ${current_money}, need ${self.food_type.buy_cost})")
                    return
                controller.buy(bot_id, self.food_type, sx, sy)
                self.buy_attempted = True
            return

        # Navigate to shop
        next_move = bfs_to_adjacent(controller, (bx, by), self.shop_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])
        else:
            print(f"[WARN] BuyIngredient: Cannot path to SHOP")

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
        super().__init__()
        self.shop_loc = None
        self.max_turns = 30  # Reduce timeout
        self.buy_attempted = False

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress
        self.check_progress(f"{bx}_{by}_{holding is not None}")

        # Can't buy if already holding something
        if holding is not None:
            if self.turn_count > 3:  # Give a few turns grace period
                print(f"[WARN] BuyPlate: Still holding item, can't buy")
            return

        if self.shop_loc is None:
            self.shop_loc = find_tile(controller, "SHOP")
            if not self.shop_loc:
                print(f"[WARN] BuyPlate: No SHOP tile found")
                return

        sx, sy = self.shop_loc

        if max(abs(bx - sx), abs(by - sy)) <= 1:
            if not self.buy_attempted:
                # Check if we can afford it
                current_money = controller.get_team_money(controller.get_team())
                if current_money < ShopCosts.PLATE.buy_cost:
                    print(f"[WARN] BuyPlate: Not enough money (have ${current_money}, need ${ShopCosts.PLATE.buy_cost})")
                    return
                controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
                self.buy_attempted = True
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.shop_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])
        else:
            print(f"[WARN] BuyPlate: Cannot path to SHOP")

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
        super().__init__()
        self.counter_loc = None
        self.state = "navigating"  # navigating → placing → chopping → picking_up → done
        self.retries = 0
        self.max_retries = 5
        self.max_turns = 40  # Reduce timeout for chopping
        self.failed = False  # Track if command failed

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress to detect stuck states
        self.check_progress(f"{self.state}_{bx}_{by}_{holding is not None}")

        # If not holding anything and not in final state, abort
        if self.state in ["navigating", "placing"] and holding is None:
            print(f"[WARN] ChopIngredient: Lost item during {self.state}, aborting")
            self.state = "done"
            self.failed = True
            return

        if self.state == "navigating":
            # Re-find counter if it's no longer valid or we've retried
            if self.counter_loc is None or self.retries > 0:
                # Find closest empty counter to current position
                self.counter_loc = find_closest_empty_tile(controller, "COUNTER", (bx, by))
                if not self.counter_loc:
                    self.retries += 1
                    print(f"[WARN] ChopIngredient: No empty counter found (retry {self.retries}/{self.max_retries})")
                    if self.retries > self.max_retries:
                        print(f"[WARN] ChopIngredient: No empty counter available after {self.max_retries} retries, aborting")
                        self.state = "done"
                        self.failed = True
                    return

            cx, cy = self.counter_loc
            if max(abs(bx - cx), abs(by - cy)) <= 1:
                # Verify counter is still empty before transitioning
                tile = controller.get_map(controller.get_team()).tiles[cx][cy]
                if getattr(tile, "item", None) is None:
                    self.state = "placing"
                else:
                    # Counter got occupied, find a new one
                    self.counter_loc = None
                    self.retries += 1
                    if self.retries > self.max_retries:
                        print(f"[WARN] ChopIngredient: Failed to find empty counter after {self.max_retries} retries, aborting")
                        self.state = "done"
                        self.failed = True
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.counter_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])
                else:
                    # Can't reach counter, find another
                    print(f"[WARN] ChopIngredient: Counter unreachable, finding alternative")
                    self.counter_loc = None
                    self.retries += 1

        elif self.state == "placing":
            if holding is not None:
                cx, cy = self.counter_loc
                controller.place(bot_id, cx, cy)
                # VERIFY: Check that we're no longer holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is None:
                    self.state = "chopping"
                else:
                    # Place failed, retry
                    print(f"[WARN] ChopIngredient: Place action failed, retrying")
                    self.state = "navigating"
                    self.counter_loc = None
                    self.retries += 1
                    if self.retries > self.max_retries:
                        print(f"[WARN] ChopIngredient: Too many place failures, aborting")
                        self.state = "done"
                        self.failed = True

        elif self.state == "chopping":
            if holding is None:
                cx, cy = self.counter_loc
                # Verify item is on counter before chopping
                tile = controller.get_map(controller.get_team()).tiles[cx][cy]
                item = getattr(tile, "item", None)
                if item and item.__class__.__name__ == "Food":
                    # Check if already chopped
                    if getattr(item, "chopped", False):
                        print(f"[INFO] ChopIngredient: Food already chopped, skipping to pickup")
                        self.state = "picking_up"
                    else:
                        controller.chop(bot_id, cx, cy)
                        # Note: chop() is instant, so we can move to pickup immediately
                        self.state = "picking_up"
                else:
                    print(f"[WARN] ChopIngredient: Expected food on counter, state desync, aborting")
                    self.state = "done"
            else:
                print(f"[WARN] ChopIngredient: Still holding item during chop state, state desync")
                self.state = "placing"

        elif self.state == "picking_up":
            cx, cy = self.counter_loc
            # Verify chopped food is still on counter
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            item = getattr(tile, "item", None)
            if item and item.__class__.__name__ == "Food" and getattr(item, "chopped", False):
                controller.pickup(bot_id, cx, cy)
                # VERIFY: Check that we're now holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is not None:
                    self.state = "done"
                else:
                    print(f"[WARN] ChopIngredient: Pickup failed, retrying")
                    self.retries += 1
                    if self.retries > self.max_retries:
                        print(f"[WARN] ChopIngredient: Too many pickup failures, aborting")
                        self.state = "done"
                        self.failed = True
            else:
                print(f"[WARN] ChopIngredient: Chopped food disappeared from counter, aborting")
                self.state = "done"
                self.failed = True

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
        super().__init__()
        self.cooker_loc = None
        self.state = "navigating"  # navigating → placing → cooking → picking_up → done
        self.max_turns = 150  # Cooking can take a while

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if self.state == "navigating":
            if self.cooker_loc is None:
                # Find closest cooker to current position
                self.cooker_loc = find_closest_tile(controller, "COOKER", (bx, by))
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
                # VERIFY: Check that we're no longer holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is None:
                    # Verify food is now in pan on cooker
                    tile = controller.get_map(controller.get_team()).tiles[cx][cy]
                    item = getattr(tile, "item", None)
                    if item and item.__class__.__name__ == "Pan":
                        self.state = "cooking"
                    else:
                        print(f"[WARN] CookIngredient: Place succeeded but no pan found, retrying")
                        self.state = "navigating"
                else:
                    print(f"[WARN] CookIngredient: Place action failed, retrying")
                    self.state = "navigating"

        elif self.state == "cooking":
            # Wait for cooking to complete (cooked_stage == 1, perfectly cooked)
            # Food is placed in a Pan on the cooker, so we need to check the Pan's food
            cx, cy = self.cooker_loc
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            item = getattr(tile, "item", None)

            # Check if there's a pan with food cooking
            if item and item.__class__.__name__ == "Pan":
                food = getattr(item, "food", None)
                if food and hasattr(food, "cooked_stage"):
                    if food.cooked_stage == 1:
                        self.state = "picking_up"
                    elif food.cooked_stage >= 2:
                        print(f"[WARN] CookIngredient: Food burned (stage {food.cooked_stage})")
                        # Food burned, abort command
                        self.state = "done"
            else:
                print(f"[WARN] CookIngredient: Pan disappeared during cooking")
                self.state = "done"

        elif self.state == "picking_up":
            cx, cy = self.cooker_loc
            # Verify food is still cooked and ready
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            item = getattr(tile, "item", None)
            if item and item.__class__.__name__ == "Pan":
                food = getattr(item, "food", None)
                if food and hasattr(food, "cooked_stage") and food.cooked_stage == 1:
                    controller.take_from_pan(bot_id, cx, cy)
                    # VERIFY: Check that we're now holding the cooked food
                    new_bot_state = controller.get_bot_state(bot_id)
                    if new_bot_state and new_bot_state.get('holding') is not None:
                        self.state = "done"
                    else:
                        print(f"[WARN] CookIngredient: take_from_pan failed")
                else:
                    print(f"[WARN] CookIngredient: Food no longer properly cooked")
                    self.state = "done"
            else:
                print(f"[WARN] CookIngredient: Pan disappeared before pickup")
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
        super().__init__()
        self.cooker_loc = None
        self.state = "navigating"  # navigating → placing → done

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if self.state == "navigating":
            if self.cooker_loc is None:
                # Find closest cooker to current position
                self.cooker_loc = find_closest_tile(controller, "COOKER", (bx, by))
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
                # VERIFY: Check that we're no longer holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is None:
                    # Verify food is now in pan on cooker
                    tile = controller.get_map(controller.get_team()).tiles[cx][cy]
                    item = getattr(tile, "item", None)
                    if item and item.__class__.__name__ == "Pan":
                        self.state = "done"
                    else:
                        print(f"[WARN] StartCooking: Place succeeded but no pan found")
                        self.state = "done"  # Still complete, just log warning
                else:
                    print(f"[WARN] StartCooking: Place action failed, retrying")
                    self.state = "navigating"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        # Complete once we've placed food in cooker
        return self.state == "done"

    def __str__(self) -> str:
        return "StartCooking()"


class FinishCookingCommand(Command):
    """Command: Pick up cooked food from cooker when ready."""

    def __init__(self, food_type: FoodType):
        super().__init__()
        self.food_type = food_type
        self.cooker_loc = None
        self.state = "waiting"  # waiting → navigating → picking_up → done
        self.max_turns = 80  # Reduce timeout - if food takes longer, it will burn
        self.food_burned = False  # Track if food burned

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
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
                if food and hasattr(food, "cooked_stage"):
                    if food.cooked_stage == 1:
                        self.state = "navigating"
                    elif food.cooked_stage >= 2:
                        print(f"[WARN] FinishCooking: Food burned (stage {food.cooked_stage})")
                        self.food_burned = True
                        self.state = "done"  # Abort, food is ruined
            else:
                # Pan disappeared, abort
                print(f"[WARN] FinishCooking: Pan disappeared while waiting")
                self.food_burned = True
                self.state = "done"

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
            # Verify food is still cooked and ready
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            item = getattr(tile, "item", None)
            if item and item.__class__.__name__ == "Pan":
                food = getattr(item, "food", None)
                if food and hasattr(food, "cooked_stage") and food.cooked_stage == 1:
                    controller.take_from_pan(bot_id, cx, cy)
                    # VERIFY: Check that we're now holding the cooked food
                    new_bot_state = controller.get_bot_state(bot_id)
                    if new_bot_state and new_bot_state.get('holding') is not None:
                        self.state = "done"
                    else:
                        print(f"[WARN] FinishCooking: take_from_pan failed")
                        self.state = "done"
                else:
                    print(f"[WARN] FinishCooking: Food no longer properly cooked")
                    self.food_burned = True
                    self.state = "done"
            else:
                print(f"[WARN] FinishCooking: Pan disappeared before pickup")
                self.food_burned = True
                self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        if self.state != "done":
            return False

        # If food burned, we're done (failed, but complete)
        if self.food_burned:
            return True

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
        super().__init__()
        self.box_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
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
        super().__init__()
        self.counter_loc = None
        self.retries = 0
        self.max_retries = 5
        self.max_turns = 40  # Reduce timeout

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress
        self.check_progress(f"{bx}_{by}_{holding is not None}")

        if holding is None:
            return

        # Re-find counter if needed or retrying
        if self.counter_loc is None:
            # Find closest empty counter to current position
            self.counter_loc = find_closest_empty_tile(controller, "COUNTER", (bx, by))
            if not self.counter_loc:
                if self.retries < self.max_retries:
                    self.retries += 1
                else:
                    print(f"[WARN] PlaceOnCounter: No empty counter found after {self.max_retries} attempts, aborting")
                return

        cx, cy = self.counter_loc

        if max(abs(bx - cx), abs(by - cy)) <= 1:
            # Verify counter is still empty before placing
            tile = controller.get_map(controller.get_team()).tiles[cx][cy]
            if getattr(tile, "item", None) is None:
                controller.place(bot_id, cx, cy)
                # VERIFY: Check that we're no longer holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is not None:
                    print(f"[WARN] PlaceOnCounter: Place action failed, retrying")
                    self.counter_loc = None
                    self.retries += 1
            else:
                # Counter got occupied, find a new one
                self.counter_loc = None
                self.retries += 1
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.counter_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])
        else:
            # Can't reach counter, find another
            print(f"[WARN] PlaceOnCounter: Counter unreachable, finding alternative")
            self.counter_loc = None
            self.retries += 1

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
        super().__init__()
        self.food_type = food_type
        self.box_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
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
        super().__init__()
        self.plate_loc = None
        self.max_turns = 40  # Reduce timeout
        self.retries = 0
        self.max_retries = 3

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress
        self.check_progress(f"{bx}_{by}_{holding is not None}")

        if holding is not None:
            return

        if self.plate_loc is None or self.retries > 0:
            # Find plate on counter (item is actual object, not dict)
            # Find closest plate to current position
            m = controller.get_map(controller.get_team())
            closest_plate = None
            min_dist = float('inf')

            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name == "COUNTER":
                        item = getattr(tile, "item", None)
                        if item and item.__class__.__name__ == "Plate":
                            dist = max(abs(x - bx), abs(y - by))
                            if dist < min_dist:
                                min_dist = dist
                                closest_plate = (x, y)

            self.plate_loc = closest_plate
            if not self.plate_loc:
                self.retries += 1
                if self.retries > self.max_retries:
                    print(f"[WARN] PickupPlateFromCounter: No plate found after {self.max_retries} attempts")
                return

        px, py = self.plate_loc

        if max(abs(bx - px), abs(by - py)) <= 1:
            # Verify plate is still there
            tile = controller.get_map(controller.get_team()).tiles[px][py]
            item = getattr(tile, "item", None)
            if item and item.__class__.__name__ == "Plate":
                controller.pickup(bot_id, px, py)
            else:
                # Plate disappeared, find another
                print(f"[WARN] PickupPlateFromCounter: Plate disappeared, searching again")
                self.plate_loc = None
                self.retries += 1
            return

        next_move = bfs_to_adjacent(controller, (bx, by), self.plate_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])
        else:
            # Can't reach plate, find another
            print(f"[WARN] PickupPlateFromCounter: Plate unreachable, finding alternative")
            self.plate_loc = None
            self.retries += 1

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
        super().__init__()
        self.box_loc = None
        self.retries = 0
        self.max_retries = 5

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        if holding is None:
            return

        # Re-find box if needed or retrying
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
                if self.retries < self.max_retries:
                    self.retries += 1
                else:
                    print(f"[WARN] PlaceInBox: No suitable box found after {self.max_retries} attempts")
                return

        box_x, box_y = self.box_loc

        if max(abs(bx - box_x), abs(by - box_y)) <= 1:
            # Verify box is still suitable before placing
            tile = controller.get_map(controller.get_team()).tiles[box_x][box_y]
            item = getattr(tile, "item", None)
            holding_type = holding.get('type') if isinstance(holding, dict) else holding.__class__.__name__

            is_suitable = (item is None or item.__class__.__name__ == holding_type)

            if is_suitable:
                controller.place(bot_id, box_x, box_y)
                # VERIFY: Check that we're no longer holding the item
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is not None:
                    print(f"[WARN] PlaceInBox: Place action failed, retrying")
                    self.box_loc = None
                    self.retries += 1
            else:
                # Box no longer suitable, find a new one
                self.box_loc = None
                self.retries += 1
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
        super().__init__()
        self.box_loc = None

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
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
        super().__init__()
        self.food_type = food_type
        self.food_loc = None
        self.state = "navigating"
        self.initial_food_count = None
        self.retries = 0
        self.max_retries = 5
        self.max_turns = 50  # Reduce timeout

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Track progress
        self.check_progress(f"{self.state}_{bx}_{by}")

        # holding is a dict
        if holding is None or holding.get('type') != 'Plate':
            print(f"[WARN] AddFoodToPlate: Not holding plate, aborting")
            self.state = "done"
            return

        if self.state == "navigating":
            # Re-find food if needed or retrying
            if self.food_loc is None:
                # Find closest food on counter
                m = controller.get_map(controller.get_team())
                closest_food = None
                min_dist = float('inf')

                for x in range(m.width):
                    for y in range(m.height):
                        tile = m.tiles[x][y]
                        if tile.tile_name == "COUNTER":
                            item = getattr(tile, "item", None)
                            if item and hasattr(item, 'food_name') and item.food_name == self.food_type.name:
                                dist = max(abs(x - bx), abs(y - by))
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_food = (x, y)

                self.food_loc = closest_food
                if not self.food_loc:
                    self.retries += 1
                    if self.retries >= self.max_retries:
                        print(f"[WARN] AddFoodToPlate: Food {self.food_type.name} not found on counter after {self.max_retries} attempts, aborting")
                        self.state = "done"
                    return

            fx, fy = self.food_loc
            if max(abs(bx - fx), abs(by - fy)) <= 1:
                # Verify food is still on counter before transitioning
                tile = controller.get_map(controller.get_team()).tiles[fx][fy]
                item = getattr(tile, "item", None)
                if item and hasattr(item, 'food_name') and item.food_name == self.food_type.name:
                    # Record initial food count on plate
                    self.initial_food_count = len(holding.get('food', []))
                    self.state = "adding"
                else:
                    # Food disappeared, find it again
                    print(f"[WARN] AddFoodToPlate: Food disappeared, searching again")
                    self.food_loc = None
                    self.retries += 1
            else:
                next_move = bfs_to_adjacent(controller, (bx, by), self.food_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])
                else:
                    # Can't reach food, find another
                    print(f"[WARN] AddFoodToPlate: Food unreachable, finding alternative")
                    self.food_loc = None
                    self.retries += 1

        elif self.state == "adding":
            fx, fy = self.food_loc
            # Verify food is still there
            tile = controller.get_map(controller.get_team()).tiles[fx][fy]
            item = getattr(tile, "item", None)
            if item and hasattr(item, 'food_name') and item.food_name == self.food_type.name:
                controller.add_food_to_plate(bot_id, fx, fy)
                # VERIFY: Check that food count increased on plate
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state:
                    new_holding = new_bot_state.get('holding')
                    if new_holding and new_holding.get('type') == 'Plate':
                        new_food_count = len(new_holding.get('food', []))
                        if new_food_count > self.initial_food_count:
                            self.state = "done"
                        else:
                            print(f"[WARN] AddFoodToPlate: Food not added to plate, retrying")
                            self.state = "navigating"
                            self.food_loc = None
                            self.retries += 1
                            if self.retries >= self.max_retries:
                                print(f"[WARN] AddFoodToPlate: Too many add failures, aborting")
                                self.state = "done"
                    else:
                        print(f"[WARN] AddFoodToPlate: Lost plate after add action, aborting")
                        self.state = "done"
            else:
                print(f"[WARN] AddFoodToPlate: Food disappeared from counter")
                self.state = "navigating"
                self.food_loc = None
                self.retries += 1
                if self.retries >= self.max_retries:
                    print(f"[WARN] AddFoodToPlate: Too many retries, aborting")
                    self.state = "done"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        return self.state == "done"

    def __str__(self) -> str:
        return f"AddFoodToPlate({self.food_type.name})"


class SubmitOrderCommand(Command):
    """Command: Submit completed plate at submit station."""

    def __init__(self, order_id: int = None):
        super().__init__()
        self.order_id = order_id  # Keep for reference, but not used in submit()
        self.submit_loc = None
        self.submitted = False

    def execute(self, controller: RobotController, bot_id: int) -> None:
        self.turn_count += 1
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
            if holding and holding.get('type') == 'Plate' and not self.submitted:
                # submit() automatically matches plate to orders
                controller.submit(bot_id, sx, sy)
                self.submitted = True
                # VERIFY: Check that we're no longer holding the plate
                new_bot_state = controller.get_bot_state(bot_id)
                if new_bot_state and new_bot_state.get('holding') is not None:
                    print(f"[WARN] SubmitOrder: Submit action failed, still holding plate")
                    self.submitted = False  # Try again
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
        if controller is not None:
            counter_count = RecipePlanner.count_tiles(controller, "COUNTER")
            box_count = RecipePlanner.count_tiles(controller, "BOX")
            # If only 1 counter and at least 1 box, use box for plate storage
            if counter_count == 1 and box_count >= 1:
                use_box_for_plate = True
                print(f"[BOT] Detected 1 counter + {box_count} box(es) - using box for plate storage")

        # Step 1: Buy plate and place in appropriate location
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
        # Decide cooking strategy based on number of non-cooking ingredients
        # If there are too many non-cooking tasks, food may burn - use sequential instead
        use_parallel_cooking = (
            cooking_foods and
            not use_box_for_plate and
            len(non_cooking_foods) <= 1  # Only parallelize if 0-1 non-cooking ingredients
        )

        if use_parallel_cooking:
            # PARALLELIZED COOKING STRATEGY (safe - only 0-1 non-cooking ingredients)
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
        self.cleanup_turns = 0  # Track how many turns we've been cleaning
        self.max_cleanup_turns = 20  # Maximum turns to spend in cleanup mode

    def has_accessible_tile_type(self, controller: RobotController, tile_name: str, from_pos: Tuple[int, int]) -> bool:
        """Check if at least one reachable tile of the given type exists on the map."""
        # Find all tiles of this type
        for x in range(self.map.width):
            for y in range(self.map.height):
                if self.map.tiles[x][y].tile_name == tile_name:
                    # Check if this tile is reachable using BFS
                    # We're checking if we can pathfind to adjacent position
                    tile_pos = (x, y)

                    # Simple check: can we path to any adjacent tile?
                    # Try to path from current position to this tile
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            adj_x, adj_y = x + dx, y + dy
                            if 0 <= adj_x < self.map.width and 0 <= adj_y < self.map.height:
                                if self.map.tiles[adj_x][adj_y].is_walkable:
                                    # Found walkable adjacent tile, check if reachable from start
                                    path = bfs_to_adjacent(controller, from_pos, tile_pos)
                                    # If path is not None, or we're already adjacent, it's reachable
                                    if path is not None or max(abs(from_pos[0] - x), abs(from_pos[1] - y)) <= 1:
                                        return True
        return False

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
        bots = controller.get_team_bot_ids(controller.get_team())
        if not bots:
            return

        bot_id = bots[0]  # Use first bot

        # Check if current order is still active (hasn't expired)
        if self.current_order_id is not None:
            orders = controller.get_orders(controller.get_team())
            order_still_active = any(o["order_id"] == self.current_order_id and o["is_active"] for o in orders)

            if not order_still_active:
                print(f"[BOT] Turn {controller.get_turn()}: Order {self.current_order_id} expired, clearing workspace")

                # Mark expired order as processed so we don't try it again
                if self.current_order_id is not None:
                    self.processed_orders.add(self.current_order_id)

                # Clear command queue and enter cleanup mode
                self.command_queue = []
                self.current_command_index = 0
                self.current_order_id = None
                self.cleaning_workspace = True  # Enter cleanup mode
                self.cleanup_turns = 0  # Reset cleanup counter
                # Cleanup logic will run on next section

        # Check if we're in cleanup mode (clearing workspace after expired order)
        if self.cleaning_workspace:
            self.cleanup_turns += 1

            # Check if we've been cleaning too long
            if self.cleanup_turns > self.max_cleanup_turns:
                print(f"[BOT] Cleanup timeout after {self.cleanup_turns} turns, exiting cleanup mode")
                self.cleaning_workspace = False
                self.cleanup_turns = 0
                return

            # Continue cleanup process
            bot_state = controller.get_bot_state(bot_id)
            if not bot_state:
                return

            bx, by = bot_state['x'], bot_state['y']
            holding = bot_state.get('holding')

            # If holding something, get rid of it
            if holding is not None:
                holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

                # Special case: clean empty plates can't be trashed (they stay in hand)
                if holding_type == 'Plate':
                    food_on_plate = holding.get('food', [])
                    is_dirty = holding.get('dirty', False)

                    if not food_on_plate and not is_dirty:
                        # Clean empty plate - place it instead of trashing
                        print(f"[BOT] Cleanup: Placing clean empty plate")
                        target_loc = find_closest_empty_tile(controller, "COUNTER", (bx, by))
                        if not target_loc:
                            target_loc = find_closest_empty_tile(controller, "BOX", (bx, by))

                        if target_loc:
                            cx, cy = target_loc
                            if max(abs(bx - cx), abs(by - cy)) <= 1:
                                controller.place(bot_id, cx, cy)
                                new_state = controller.get_bot_state(bot_id)
                                if new_state and new_state.get('holding') is None:
                                    print(f"[BOT] Placed clean plate during cleanup")
                                return
                            else:
                                next_move = bfs_to_adjacent(controller, (bx, by), target_loc)
                                if next_move:
                                    controller.move(bot_id, next_move[0], next_move[1])
                                return
                        else:
                            # No space - just exit cleanup
                            print(f"[BOT] No space for plate, exiting cleanup")
                            self.cleaning_workspace = False
                            return

                # For everything else (food, dirty plates, pans), trash it
                trash_loc = find_tile(controller, "TRASH")
                if trash_loc:
                    tx, ty = trash_loc
                    if max(abs(bx - tx), abs(by - ty)) <= 1:
                        controller.trash(bot_id, tx, ty)
                        print(f"[BOT] Trashed {holding_type} during cleanup")
                        return
                    else:
                        next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                        if next_move:
                            controller.move(bot_id, next_move[0], next_move[1])
                        return
                else:
                    # No trash available, just exit cleanup mode
                    print(f"[BOT] No trash tile found, exiting cleanup")
                    self.cleaning_workspace = False
                    return

            # Hands are empty, clean up items from counters (prioritize Food over Plates)
            m = controller.get_map(controller.get_team())
            cleanup_target = None

            # First pass: look for Food items
            for x in range(m.width):
                for y in range(m.height):
                    tile = m.tiles[x][y]
                    if tile.tile_name in ["COUNTER", "BOX"]:
                        item = getattr(tile, "item", None)
                        if item is not None and item.__class__.__name__ == "Food":
                            cleanup_target = (x, y)
                            break
                if cleanup_target:
                    break

            # Second pass: if no food found, look for Plates
            if not cleanup_target:
                for x in range(m.width):
                    for y in range(m.height):
                        tile = m.tiles[x][y]
                        if tile.tile_name in ["COUNTER", "BOX"]:
                            item = getattr(tile, "item", None)
                            if item is not None and item.__class__.__name__ == "Plate":
                                cleanup_target = (x, y)
                                break
                    if cleanup_target:
                        break

            if cleanup_target:
                cx, cy = cleanup_target
                if max(abs(bx - cx), abs(by - cy)) <= 1:
                    controller.pickup(bot_id, cx, cy)
                    item_type = m.tiles[cx][cy].item.__class__.__name__ if m.tiles[cx][cy].item else "item"
                    print(f"[BOT] Picked up {item_type} from {m.tiles[cx][cy].tile_name} to clear workspace")
                    return
                else:
                    next_move = bfs_to_adjacent(controller, (bx, by), cleanup_target)
                    if next_move:
                        controller.move(bot_id, next_move[0], next_move[1])
                    return

            # Workspace is clean
            print(f"[BOT] Workspace cleared, ready for new orders")
            self.cleaning_workspace = False
            return

        # If no commands, check for new orders
        # BUT: only if bot has empty hands (don't start new orders while holding items from expired orders)
        if not self.command_queue or self.current_command_index >= len(self.command_queue):
            bot_state = controller.get_bot_state(bot_id)
            if bot_state and bot_state.get('holding') is not None:
                # Still holding something from a previous order
                holding = bot_state.get('holding')
                holding_type = holding.get('type') if isinstance(holding, dict) else 'unknown'

                # Special case: If holding a clean empty plate, place it somewhere
                # (trashing plates makes them clean but doesn't remove them, causing infinite loops)
                if holding_type == 'Plate':
                    food_on_plate = holding.get('food', [])
                    is_dirty = holding.get('dirty', False)

                    # If it's a clean empty plate, we need to place it (can't trash it)
                    if not food_on_plate and not is_dirty:
                        print(f"[BOT] Holding clean empty plate, placing on counter/box")

                        # Try to find an empty spot (closest to current position)
                        bx, by = bot_state['x'], bot_state['y']
                        target_loc = find_closest_empty_tile(controller, "COUNTER", (bx, by))
                        if not target_loc:
                            target_loc = find_closest_empty_tile(controller, "BOX", (bx, by))

                        if target_loc:
                            cx, cy = target_loc
                            bx, by = bot_state['x'], bot_state['y']

                            if max(abs(bx - cx), abs(by - cy)) <= 1:
                                controller.place(bot_id, cx, cy)
                                # Verify we actually placed it
                                new_state = controller.get_bot_state(bot_id)
                                if new_state and new_state.get('holding') is None:
                                    print(f"[BOT] Placed clean plate successfully")
                                else:
                                    print(f"[BOT] Failed to place clean plate")
                            else:
                                next_move = bfs_to_adjacent(controller, (bx, by), target_loc)
                                if next_move:
                                    controller.move(bot_id, next_move[0], next_move[1])
                        else:
                            # No empty tiles - just accept we're holding a plate and continue
                            print(f"[BOT] No empty tiles for clean plate, will continue with new orders")
                        return
                    else:
                        # Dirty plate or plate with food - trash it
                        print(f"[BOT] Holding dirty/full plate, trashing it")
                        trash_loc = find_tile(controller, "TRASH")
                        if trash_loc:
                            tx, ty = trash_loc
                            bx, by = bot_state['x'], bot_state['y']

                            if max(abs(bx - tx), abs(by - ty)) <= 1:
                                controller.trash(bot_id, tx, ty)
                                # After trashing, the plate becomes clean and empty
                                # On next turn, we'll place it
                            else:
                                next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                                if next_move:
                                    controller.move(bot_id, next_move[0], next_move[1])
                        return
                else:
                    # For non-plate items (food, pan), trash them
                    print(f"[BOT] Holding {holding_type} between orders, trashing it")
                    trash_loc = find_tile(controller, "TRASH")
                    if trash_loc:
                        tx, ty = trash_loc
                        bx, by = bot_state['x'], bot_state['y']

                        if max(abs(bx - tx), abs(by - ty)) <= 1:
                            controller.trash(bot_id, tx, ty)
                            print(f"[BOT] Trashed {holding_type}")
                        else:
                            next_move = bfs_to_adjacent(controller, (bx, by), trash_loc)
                            if next_move:
                                controller.move(bot_id, next_move[0], next_move[1])
                return

            orders = controller.get_orders(controller.get_team())
            current_turn = controller.get_turn()

            # Get bot position for accessibility checks
            bot_state = controller.get_bot_state(bot_id)
            if not bot_state:
                return
            bot_pos = (bot_state['x'], bot_state['y'])

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
                    chopping_count = sum(1 for food_name in order["required"]
                                        if food_name in [ft.name for ft in FoodType]
                                        and getattr(FoodType[food_name], 'can_chop', False))

                    # RESOURCE AVAILABILITY CHECK: Skip orders requiring inaccessible resources
                    if cooking_count > 0 and not self.has_accessible_tile_type(controller, "COOKER", bot_pos):
                        print(f"[BOT] Skipping order {order['order_id']} - requires cooking but no accessible COOKER tiles")
                        self.processed_orders.add(order["order_id"])
                        continue

                    if chopping_count > 0 and not self.has_accessible_tile_type(controller, "COUNTER", bot_pos):
                        print(f"[BOT] Skipping order {order['order_id']} - requires chopping but no accessible COUNTER tiles")
                        self.processed_orders.add(order["order_id"])
                        continue

                    # Balanced formula - realistic estimates with safety margin
                    base_turns = 25  # Buy plate, setup, submit, navigation overhead
                    ingredient_turns = num_ingredients * 10  # Each ingredient: buy + navigate + process + add to plate
                    chopping_turns = chopping_count * 7  # Chopping: place, chop, pickup

                    # Cooking time - account for waiting and parallelization
                    if cooking_count > 0:
                        # If we can parallelize (0-1 non-cooking ingredients), first cook happens during other prep
                        can_parallelize = (num_ingredients - cooking_count) <= 1
                        if can_parallelize:
                            # First cook overlaps with prep, but still costs time
                            cooking_turns = max(0, cooking_count - 1) * 28 + 18
                        else:
                            # All cooking is sequential - each cook takes significant time
                            cooking_turns = cooking_count * 28
                    else:
                        cooking_turns = 0

                    estimated_turns = base_turns + ingredient_turns + chopping_turns + cooking_turns

                    # Skip orders where we don't have enough time (need 115% buffer - balanced)
                    if remaining_turns < estimated_turns * 1.15:
                        print(f"[BOT] Skipping order {order['order_id']} - too complex ({num_ingredients} ingredients, {cooking_count} cooking) "
                              f"for {remaining_turns} turns (need ~{int(estimated_turns * 1.15)})")
                        self.processed_orders.add(order["order_id"])
                        continue

                    # Calculate value (reward + penalty)
                    reward = order.get('reward', 0)
                    penalty = order.get('penalty', 0)
                    total_value = reward + penalty

                    # PROFITABILITY CHECK: Calculate ingredient cost
                    ingredient_cost = 0
                    for food_name in order["required"]:
                        if food_name in [ft.name for ft in FoodType]:
                            food_type = FoodType[food_name]
                            ingredient_cost += int(food_type.buy_cost)
                    ingredient_cost += int(ShopCosts.PLATE.buy_cost)  # Add plate cost

                    # Skip unprofitable orders (cost >= value)
                    if ingredient_cost >= total_value:
                        print(f"[BOT] Skipping order {order['order_id']} - UNPROFITABLE (cost=${ingredient_cost} >= value=${total_value})")
                        self.processed_orders.add(order["order_id"])
                        continue

                    # Calculate profit and profit per turn (efficiency)
                    profit = total_value - ingredient_cost
                    profit_per_turn = profit / estimated_turns

                    candidate_orders.append({
                        'order': order,
                        'expires_turn': expires_turn,
                        'remaining_turns': remaining_turns,
                        'estimated_turns': estimated_turns,
                        'total_value': total_value,
                        'reward': reward,
                        'penalty': penalty,
                        'cost': ingredient_cost,
                        'profit': profit,
                        'profit_per_turn': profit_per_turn
                    })

            if not candidate_orders:
                return

            # Debug: Show all candidate orders
            print(f"[BOT] Evaluating {len(candidate_orders)} candidate orders:")
            for c in candidate_orders:
                print(f"  Order {c['order']['order_id']}: profit=${c['profit']}, "
                      f"est_turns={int(c['estimated_turns'])}, efficiency={c['profit_per_turn']:.2f} profit/turn")

            # Strategy: Adaptive selection based on remaining game time
            # When time is limited, prefer quick profitable orders over slow efficient ones
            game_turns_remaining = 500 - current_turn

            if game_turns_remaining < 200:
                # Late game: Prioritize SPEED and good profit - complete more orders quickly
                # Sort by: speed (primary) then profit (secondary) - favor fast completions
                candidate_orders.sort(key=lambda c: (c['estimated_turns'], -c['profit']))
                print(f"[BOT] Late game strategy: prioritizing speed + profit")
                buffer_multiplier = 1.30  # Very strict buffer in late game to ensure completion
            else:
                # Early/mid game: Maximize efficiency (profit per turn)
                candidate_orders.sort(key=lambda c: c['profit_per_turn'], reverse=True)
                print(f"[BOT] Early/mid game strategy: prioritizing efficiency")
                buffer_multiplier = 1.15  # Standard buffer

            # Choose the best order that we have time to complete
            best = None
            for candidate in candidate_orders:
                # Use adaptive buffer based on game phase
                if candidate['remaining_turns'] >= candidate['estimated_turns'] * buffer_multiplier:
                    best = candidate
                    break

            # If no order passes the time check, don't start any order (avoid wasting money)
            if best is None:
                print(f"[BOT] No orders have enough time remaining (need {int(buffer_multiplier*100)}% buffer)")
                return

            order = best['order']
            print(f"[BOT] Turn {current_turn}: Processing order {order['order_id']}: {order['required']} "
                  f"({best['remaining_turns']} turns remaining, est={int(best['estimated_turns'])} turns, "
                  f"profit=${best['profit']}, efficiency=${best['profit_per_turn']:.2f} profit/turn)")

            # DON'T add to processed_orders yet - only mark as processed when successfully completed
            # self.processed_orders.add(order["order_id"])
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

        # Log current command state every 10 turns
        if current_cmd.turn_count % 10 == 0 and current_cmd.turn_count > 0:
            print(f"[BOT] Still executing {current_cmd} (turn {current_cmd.turn_count})")
            # Log specific state for commands with internal state
            if hasattr(current_cmd, 'state'):
                print(f"[BOT]   State: {current_cmd.state}")
            if hasattr(current_cmd, 'retries'):
                print(f"[BOT]   Retries: {current_cmd.retries}")

        # Check if command is stuck (timeout)
        if current_cmd.is_stuck():
            print(f"[BOT] Command STUCK (timeout after {current_cmd.turn_count} turns): {current_cmd}")
            if hasattr(current_cmd, 'state'):
                print(f"[BOT]   Final state: {current_cmd.state}")
            if hasattr(current_cmd, 'stuck_counter'):
                print(f"[BOT]   Stuck counter: {current_cmd.stuck_counter}")
            print(f"[BOT] Aborting current order {self.current_order_id} due to stuck command")

            # Mark order as processed so we don't try again
            if self.current_order_id is not None:
                self.processed_orders.add(self.current_order_id)

            # Clear command queue and enter cleanup mode
            self.command_queue = []
            self.current_command_index = 0
            self.current_order_id = None
            self.cleaning_workspace = True
            self.cleanup_turns = 0  # Reset cleanup counter
            return

        # Execute
        current_cmd.execute(controller, bot_id)

        # Check completion
        if current_cmd.is_complete(controller, bot_id):
            print(f"[BOT] Command complete: {current_cmd}")
            current_cmd.on_complete(controller, bot_id)
            self.advance_command()

            # If we just finished the last command, mark order as successfully processed
            if self.current_command_index >= len(self.command_queue):
                if self.current_order_id is not None:
                    print(f"[BOT] Order {self.current_order_id} successfully completed!")
                    self.processed_orders.add(self.current_order_id)
                self.current_order_id = None
