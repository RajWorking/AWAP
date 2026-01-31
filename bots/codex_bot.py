from collections import deque
from typing import Dict, List, Optional, Tuple

from game_constants import FoodType, ShopCosts
from robot_controller import RobotController
from item import Food, Plate, Pan


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.tile_positions: Dict[str, List[Tuple[int, int]]] = {}
        self._scan_tiles()

        self.mode: Optional[str] = None
        self.noodle_state: Dict[int, int] = {}

    def _scan_tiles(self):
        self.tile_positions = {}
        for x in range(self.map.width):
            for y in range(self.map.height):
                name = getattr(self.map.tiles[x][y], "tile_name", "")
                self.tile_positions.setdefault(name, []).append((x, y))

    def _nearest(self, positions, x, y):
        if not positions:
            return None
        best = None
        best_dist = 10**9
        for px, py in positions:
            dist = max(abs(px - x), abs(py - y))
            if dist < best_dist:
                best_dist = dist
                best = (px, py)
        return best

    def _bfs_step(self, controller: RobotController, start, goal_fn):
        m = controller.get_map(controller.get_team())
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (cx, cy), path = queue.popleft()
            if goal_fn(cx, cy):
                return (0, 0) if not path else path[0]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited:
                        continue
                    if 0 <= nx < m.width and 0 <= ny < m.height and m.is_tile_walkable(nx, ny):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def _move_to_adjacent(self, controller: RobotController, bot_id: int, tx: int, ty: int) -> bool:
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return False
        bx, by = bot["x"], bot["y"]
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return True
        blocked = set()
        for other_id in controller.get_team_bot_ids(controller.get_team()):
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other:
                blocked.add((other["x"], other["y"]))
        m = controller.get_map(controller.get_team())
        def goal_fn(x, y):
            return max(abs(x - tx), abs(y - ty)) <= 1
        queue = deque([((bx, by), [])])
        visited = {(bx, by)}
        step = None
        while queue:
            (cx, cy), path = queue.popleft()
            if goal_fn(cx, cy):
                step = (0, 0) if not path else path[0]
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    if 0 <= nx < m.width and 0 <= ny < m.height and m.is_tile_walkable(nx, ny):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(dx, dy)]))
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
        return False

    def _choose_assembly_counter(self):
        submit_positions = self.tile_positions.get("SUBMIT", [])
        counter_positions = self.tile_positions.get("COUNTER", [])
        if not counter_positions:
            return None
        if not submit_positions:
            return counter_positions[0]
        best = None
        best_dist = 10**9
        for c in counter_positions:
            dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in submit_positions)
            if dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _active_orders(self, controller: RobotController):
        orders = controller.get_orders(controller.get_team())
        return [o for o in orders if o.get("is_active")]

    def _detect_mode(self, controller: RobotController):
        orders = self._active_orders(controller)
        if not orders:
            return None
        if len(orders) == 1:
            req = [r.upper() for r in orders[0].get("required", [])]
            if sorted(req) == ["MEAT", "NOODLES"]:
                return "noodle_meat"
        for o in orders:
            req = [r.upper() for r in o.get("required", [])]
            if req and all(r == "SAUCE" for r in req):
                return "sauce_only"
        return "generic"

    # ---- Noodle + Meat specialist ----
    def _play_noodle_meat(self, controller: RobotController, bot_id: int):
        state = self.noodle_state.get(bot_id, 0)
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return
        bx, by = bot["x"], bot["y"]

        counter_pos = self._nearest(self.tile_positions.get("COUNTER", []), bx, by)
        cooker_pos = self._nearest(self.tile_positions.get("COOKER", []), bx, by)
        shop_pos = self._nearest(self.tile_positions.get("SHOP", []), bx, by)
        submit_pos = self._nearest(self.tile_positions.get("SUBMIT", []), bx, by)
        trash_pos = self._nearest(self.tile_positions.get("TRASH", []), bx, by)

        if not counter_pos or not cooker_pos or not shop_pos or not submit_pos:
            return

        cx, cy = counter_pos
        kx, ky = cooker_pos
        sx, sy = shop_pos
        ux, uy = submit_pos

        holding = bot.get("holding")
        if holding and holding.get("type") == "Plate" and holding.get("dirty") and trash_pos:
            tx, ty = trash_pos
            if self._move_to_adjacent(controller, bot_id, tx, ty):
                controller.trash(bot_id, tx, ty)
            return

        if state == 0:
            self.noodle_state[bot_id] = 1
            return

        if state == 1:
            if holding:
                self.noodle_state[bot_id] = 2
                return
            if self._move_to_adjacent(controller, bot_id, sx, sy):
                if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                    self.noodle_state[bot_id] = 2
            return

        if state == 2:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 3
            return

        if state == 3:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.chop(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 4
            return

        if state == 4:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 5
            return

        if state == 5:
            if self._move_to_adjacent(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    self.noodle_state[bot_id] = 6
            return

        if state == 6:
            if holding:
                self.noodle_state[bot_id] = 7
                return
            if self._move_to_adjacent(controller, bot_id, sx, sy):
                if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                    self.noodle_state[bot_id] = 7
            return

        if state == 7:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 8
            return

        if state == 8:
            if holding:
                self.noodle_state[bot_id] = 9
                return
            if self._move_to_adjacent(controller, bot_id, sx, sy):
                if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                    self.noodle_state[bot_id] = 9
            return

        if state == 9:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 10
            return

        if state == 10:
            if self._move_to_adjacent(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                pan = getattr(tile, "item", None)
                food = pan.food if isinstance(pan, Pan) else None
                if isinstance(food, Food) and food.cooked_stage == 1:
                    if controller.take_from_pan(bot_id, kx, ky):
                        self.noodle_state[bot_id] = 11
            return

        if state == 11:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 12
            return

        if state == 12:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.noodle_state[bot_id] = 13
            return

        if state == 13:
            if self._move_to_adjacent(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    self.noodle_state[bot_id] = 0
            return

    # ---- Sauce-only handler ----
    def _play_sauce_only(self, controller: RobotController, bot_id: int, target_order):
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return
        bx, by = bot["x"], bot["y"]
        counter_pos = self._choose_assembly_counter()
        shop_pos = self._nearest(self.tile_positions.get("SHOP", []), bx, by)
        submit_pos = self._nearest(self.tile_positions.get("SUBMIT", []), bx, by)
        if not counter_pos or not shop_pos or not submit_pos:
            return
        cx, cy = counter_pos
        sx, sy = shop_pos
        ux, uy = submit_pos

        tile = controller.get_tile(controller.get_team(), cx, cy)
        plate = tile.item if tile and isinstance(getattr(tile, "item", None), Plate) else None
        holding = bot.get("holding")

        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            trash_pos = self._nearest(self.tile_positions.get("TRASH", []), bx, by)
            if trash_pos and self._move_to_adjacent(controller, bot_id, trash_pos[0], trash_pos[1]):
                controller.trash(bot_id, trash_pos[0], trash_pos[1])
            return

        if plate is None and holding is None:
            if self._move_to_adjacent(controller, bot_id, sx, sy):
                controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
            return

        if holding and holding.get("type") == "Plate" and plate is None:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                controller.place(bot_id, cx, cy)
            return

        if plate is None:
            return

        needed_count = len(target_order.get("required", []))
        current_count = 0
        for f in plate.food:
            if isinstance(f, Food) and f.food_name == "SAUCE":
                current_count += 1

        if current_count >= needed_count:
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    if self._move_to_adjacent(controller, bot_id, ux, uy):
                        controller.submit(bot_id, ux, uy)
            return

        if holding and holding.get("type") == "Food":
            if self._move_to_adjacent(controller, bot_id, cx, cy):
                controller.add_food_to_plate(bot_id, cx, cy)
            return

        if holding is None:
            if self._move_to_adjacent(controller, bot_id, sx, sy):
                controller.buy(bot_id, FoodType.SAUCE, sx, sy)
            return

    def play_turn(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return

        if self.mode is None:
            self.mode = self._detect_mode(controller)

        if self.mode == "noodle_meat":
            self._play_noodle_meat(controller, bot_ids[0])
            for bid in bot_ids[1:]:
                bot = controller.get_bot_state(bid)
                if not bot:
                    continue
                bx, by = bot["x"], bot["y"]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if controller.get_map(controller.get_team()).is_tile_walkable(bx + dx, by + dy):
                        controller.move(bid, dx, dy)
                        break
            return

        if self.mode == "sauce_only":
            orders = self._active_orders(controller)
            target = None
            for o in orders:
                req = [r.upper() for r in o.get("required", [])]
                if req and all(r == "SAUCE" for r in req):
                    target = o
                    break
            if target:
                self._play_sauce_only(controller, bot_ids[0], target)
            return

        if self.mode == "generic":
            self._play_noodle_meat(controller, bot_ids[0])
