from collections import deque
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod

from robot_controller import RobotController
from game_constants import FoodType, ShopCosts


# ============================================================================
# TASK FRAMEWORK
# ============================================================================

class Task(ABC):
    """Base class for all bot tasks."""

    @abstractmethod
    def execute(self, controller: RobotController, bot_id: int) -> None:
        """Execute one step of this task."""
        pass

    @abstractmethod
    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Check if this task is complete. If so, bot moves to next task."""
        pass

    def on_start(self, controller: RobotController, bot_id: int) -> None:
        """Called when task becomes active (optional)."""
        pass

    def on_complete(self, controller: RobotController, bot_id: int) -> None:
        """Called when task completes (optional)."""
        pass


class GoToShopAndBuy(Task):
    """Task: Walk to shop and buy a specific item."""

    def __init__(self, item_type):
        self.item_type = item_type  # FoodType enum or ShopCosts enum
        self.shop_loc = None
        self.bought = False

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']

        # Find shop on first turn
        if self.shop_loc is None:
            self.shop_loc = self._find_tile(controller, "SHOP")
            if not self.shop_loc:
                return

        sx, sy = self.shop_loc

        # If adjacent to shop, buy the item
        if max(abs(bx - sx), abs(by - sy)) <= 1:
            self._try_buy(controller, bot_id, sx, sy)
            return

        # Otherwise, move toward shop
        next_move = self._bfs_to_goal(controller, (bx, by), self.shop_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Task complete when bot is holding the desired item type."""
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        if not holding:
            return False
        # Check if holding the right type
        return holding.get('type') == 'Food' or holding.get('type') == 'Plate' or holding.get('type') == 'Pan'

    def _find_tile(self, controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].tile_name == tile_name:
                    return (x, y)
        return None

    def _try_buy(self, controller: RobotController, bot_id: int, sx: int, sy: int) -> None:
        tile = controller.get_tile(controller.get_team(), sx, sy)
        shop_items = getattr(tile, "shop_items", None)
        choice = None

        if shop_items:
            # Try to find the requested item
            for it in shop_items:
                if it == self.item_type:
                    choice = it
                    break
            # Fallback: pick first FoodType available
            if choice is None:
                for it in shop_items:
                    if isinstance(it, FoodType):
                        choice = it
                        break

        if choice is not None:
            controller.buy(bot_id, choice, sx, sy)
            self.bought = True

    def _bfs_to_goal(self, controller: RobotController, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS to find next step toward goal, avoiding occupied tiles."""
        from collections import deque
        
        queue = deque([(start, [])])
        visited = {start}
        m = controller.get_map()
        
        # Build occupied set from known bot ids
        occupied = set()
        for probe_id in range(0, 16):
            st = controller.get_bot_state(probe_id)
            if st:
                occupied.add((st['x'], st['y']))

        while queue:
            (x, y), path = queue.popleft()

            # If current position is adjacent to the goal, we've reached a valid approach
            gx, gy = goal
            if max(abs(x - gx), abs(y - gy)) <= 1:
                return path[0] if path else None

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in visited:
                        # avoid stepping on occupied tiles (except starting tile)
                        if (nx, ny) in occupied and (nx, ny) != start:
                            continue
                        if m.tiles[nx][ny].is_walkable:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None


class PlaceItemOnCounter(Task):
    """Task: Find nearest counter and place currently held item on it."""

    def __init__(self):
        self.counter_loc = None
        self.placed = False

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        # If not holding anything, task is complete
        holding = bot_state.get('holding')
        if holding is None:
            return

        bx, by = bot_state['x'], bot_state['y']

        # Find counter on first turn
        if self.counter_loc is None:
            self.counter_loc = self._find_tile(controller, "COUNTER")
            if not self.counter_loc:
                return

        cx, cy = self.counter_loc

        # If adjacent to counter, place the item
        if max(abs(bx - cx), abs(by - cy)) <= 1:
            controller.place(bot_id, cx, cy)
            self.placed = True
            return

        # Otherwise, move toward counter
        next_move = self._bfs_to_goal(controller, (bx, by), self.counter_loc)
        if next_move:
            controller.move(bot_id, next_move[0], next_move[1])

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Task complete when bot is no longer holding anything."""
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        holding = bot_state.get('holding')
        return holding is None

    def _find_tile(self, controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].tile_name == tile_name:
                    return (x, y)
        return None

    def _bfs_to_goal(self, controller: RobotController, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS to find next step toward goal, avoiding occupied tiles."""
        queue = deque([(start, [])])
        visited = {start}
        m = controller.get_map()
        
        # Build occupied set from known bot ids
        occupied = set()
        for probe_id in range(0, 16):
            st = controller.get_bot_state(probe_id)
            if st:
                occupied.add((st['x'], st['y']))

        while queue:
            (x, y), path = queue.popleft()

            # If current position is adjacent to the goal, we've reached a valid approach
            gx, gy = goal
            if max(abs(x - gx), abs(y - gy)) <= 1:
                return path[0] if path else None

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in visited:
                        # avoid stepping on occupied tiles (except starting tile)
                        if (nx, ny) in occupied and (nx, ny) != start:
                            continue
                        if m.tiles[nx][ny].is_walkable:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None


class ChopItemOnCounter(Task):
    """Task: Walk to counter, place item, chop it, and pick it back up."""

    def __init__(self):
        self.counter_loc = None
        self.state = "walking"  # walking, placing, chopping, picking_up, complete

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Find counter on first turn
        if self.counter_loc is None:
            self.counter_loc = self._find_tile(controller, "COUNTER")
            if not self.counter_loc:
                return

        cx, cy = self.counter_loc

        # State machine
        if self.state == "walking":
            # If adjacent to counter, transition to placing
            if max(abs(bx - cx), abs(by - cy)) <= 1:
                self.state = "placing"
            else:
                # Move toward counter
                next_move = self._bfs_to_goal(controller, (bx, by), self.counter_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

        elif self.state == "placing":
            # Place the item on the counter
            if holding is not None:
                controller.place(bot_id, cx, cy)
                self.state = "chopping"

        elif self.state == "chopping":
            # Chop the item on the counter (we must be holding nothing)
            holding_now = controller.get_bot_state(bot_id).get('holding')
            if holding_now is None:
                controller.chop(bot_id, cx, cy)
                self.state = "picking_up"

        elif self.state == "picking_up":
            # Pick up the chopped item
            controller.pickup(bot_id, cx, cy)
            self.state = "complete"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Task complete when we've chopped and picked back up."""
        return self.state == "complete" and controller.get_bot_state(bot_id).get('holding') is not None

    def _find_tile(self, controller: RobotController, tile_name: str) -> Optional[Tuple[int, int]]:
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].tile_name == tile_name:
                    return (x, y)
        return None

    def _bfs_to_goal(self, controller: RobotController, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS to find next step toward goal, avoiding occupied tiles."""
        queue = deque([(start, [])])
        visited = {start}
        m = controller.get_map()
        
        # Build occupied set from known bot ids
        occupied = set()
        for probe_id in range(0, 16):
            st = controller.get_bot_state(probe_id)
            if st:
                occupied.add((st['x'], st['y']))

        while queue:
            (x, y), path = queue.popleft()

            # If current position is adjacent to the goal, we've reached a valid approach
            gx, gy = goal
            if max(abs(x - gx), abs(y - gy)) <= 1:
                return path[0] if path else None

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in visited:
                        # avoid stepping on occupied tiles (except starting tile)
                        if (nx, ny) in occupied and (nx, ny) != start:
                            continue
                        if m.tiles[nx][ny].is_walkable:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None


class AddFoodToPlate(Task):
    """Task: Bot holds a plate, finds food on counter, and adds it to the plate."""

    def __init__(self, food_type: Optional[FoodType] = None):
        self.food_type = food_type  # FoodType enum or None to find any food
        self.food_loc = None
        self.state = "finding_food"  # finding_food, moving, adding, complete

    def execute(self, controller: RobotController, bot_id: int) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return

        bx, by = bot_state['x'], bot_state['y']
        holding = bot_state.get('holding')

        # Must be holding a plate
        if holding is None or holding.get('type') != 'Plate':
            return

        # Find food on counter on first turn
        if self.food_loc is None:
            self.food_loc = self._find_food_on_counter(controller)
            if not self.food_loc:
                self.state = "complete"  # No food found, mark complete
                return

        fx, fy = self.food_loc

        # State machine
        if self.state == "finding_food":
            # If adjacent to food, transition to adding
            if max(abs(bx - fx), abs(by - fy)) <= 1:
                self.state = "adding"
            else:
                # Move toward food
                next_move = self._bfs_to_goal(controller, (bx, by), self.food_loc)
                if next_move:
                    controller.move(bot_id, next_move[0], next_move[1])

        elif self.state == "adding":
            # Add food to plate (bot holds plate, targets food on counter)
            controller.add_food_to_plate(bot_id, fx, fy)
            self.state = "complete"

    def is_complete(self, controller: RobotController, bot_id: int) -> bool:
        """Task complete when we've attempted to add food."""
        return self.state == "complete"

    def _find_food_on_counter(self, controller: RobotController) -> Optional[Tuple[int, int]]:
        """Find food item on a counter tile matching the specified food_type."""
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COUNTER":
                    # Check if tile has an item
                    item = getattr(tile, "item", None)
                    if item and getattr(item, "type", None) == "Food":
                        # If food_type is specified, match it; otherwise accept any food
                        if self.food_type is None:
                            return (x, y)
                        elif getattr(item, "food_type", None) == self.food_type:
                            return (x, y)
        return None

    def _bfs_to_goal(self, controller: RobotController, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS to find next step toward goal, avoiding occupied tiles."""
        queue = deque([(start, [])])
        visited = {start}
        m = controller.get_map()
        
        # Build occupied set from known bot ids
        occupied = set()
        for probe_id in range(0, 16):
            st = controller.get_bot_state(probe_id)
            if st:
                occupied.add((st['x'], st['y']))

        while queue:
            (x, y), path = queue.popleft()

            # If current position is adjacent to the goal, we've reached a valid approach
            gx, gy = goal
            if max(abs(x - gx), abs(y - gy)) <= 1:
                return path[0] if path else None

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in visited:
                        # avoid stepping on occupied tiles (except starting tile)
                        if (nx, ny) in occupied and (nx, ny) != start:
                            continue
                        if m.tiles[nx][ny].is_walkable:
                            visited.add((nx, ny))
                            queue.append(((nx, ny), path + [(dx, dy)]))
        return None


# ============================================================================
# BOT IMPLEMENTATION
# ============================================================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        # Task queue: list of tasks to execute in order
        self.task_queue: List[Task] = [
            GoToShopAndBuy(FoodType.NOODLES),
            PlaceItemOnCounter(),#closest counter, not always the same one
            GoToShopAndBuy(ShopCosts.PLATE),
            AddFoodToPlate(FoodType.NOODLES),#closest counter, not always the same one
            PlaceItemOnCounter(),#closest counter, not always the same one
            GoToShopAndBuy(FoodType.MEAT),
            ChopItemOnCounter()
        ]
        self.current_task_index = 0

    def get_current_task(self) -> Optional[Task]:
        """Get the currently active task, or None if all tasks complete."""
        if self.current_task_index >= len(self.task_queue):
            return None
        return self.task_queue[self.current_task_index]

    def advance_task(self) -> None:
        """Move to the next task in the queue."""
        self.current_task_index += 1

    def play_turn(self, controller: RobotController):
        bots = controller.get_team_bot_ids()
        if not bots:
            return

        bot_id = bots[0]
        task = self.get_current_task()

        if task is None:
            # All tasks complete; just idle
            return

        # Execute the current task
        task.execute(controller, bot_id)

        # Check if task is complete
        if task.is_complete(controller, bot_id):
            print(f"[BOT] Task {task.__class__.__name__} complete, advancing")
            task.on_complete(controller, bot_id)
            self.advance_task()
