from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

from game_constants import FoodType, ShopCosts
from robot_controller import RobotController
from item import Food, Plate, Pan


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.tiles: Dict[str, List[Tuple[int, int]]] = {}
        self._scan_tiles()

        self.assembly_counter: Optional[Tuple[int, int]] = None
        self.prep_counter: Optional[Tuple[int, int]] = None
        self.cooker_tile: Optional[Tuple[int, int]] = None

        self.role_by_bot: Dict[int, str] = {}
        self.turn_initialized = False
        self.current_order_id: Optional[int] = None

        self.single_plan: List[Dict] = []
        self.single_plan_idx = 0
        self.single_processed_orders = set()
        self._seen_switch = False

    def _refresh_from_controller(self, controller: RobotController, bot_pos: Optional[Tuple[int, int]] = None):
        info = controller.get_switch_info()
        if info.get("my_team_switched") and not self._seen_switch:
            self._seen_switch = True
            self.map = controller.get_map(controller.get_team())
            self._scan_tiles()
            self.assembly_counter = None
            self.prep_counter = None
            self.cooker_tile = None
            self.turn_initialized = False
            return

        if self.map.width != controller.get_map(controller.get_team()).width or self.map.height != controller.get_map(controller.get_team()).height:
            self.map = controller.get_map(controller.get_team())
            self._scan_tiles()
            self.assembly_counter = None
            self.prep_counter = None
            self.cooker_tile = None
            self.turn_initialized = False

    def _reachable_walkable(self, controller: RobotController, start: Tuple[int, int]):
        m = controller.get_map(controller.get_team())
        if not m.in_bounds(start[0], start[1]) or not m.is_tile_walkable(start[0], start[1]):
            return set()
        queue = deque([start])
        visited = {start}
        while queue:
            cx, cy = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited:
                        continue
                    if 0 <= nx < m.width and 0 <= ny < m.height and m.is_tile_walkable(nx, ny):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    def _target_accessible(self, controller: RobotController, target: Tuple[int, int], reachable: set) -> bool:
        m = controller.get_map(controller.get_team())
        tx, ty = target
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < m.width and 0 <= ny < m.height and m.is_tile_walkable(nx, ny):
                    if (nx, ny) in reachable:
                        return True
        return False

    def _choose_accessible(self, controller: RobotController, positions, start: Tuple[int, int]):
        if not positions:
            return None
        reachable = self._reachable_walkable(controller, start)
        accessible = [p for p in positions if self._target_accessible(controller, p, reachable)]
        candidates = accessible if accessible else positions
        return self._nearest(candidates, start[0], start[1])

    # ----------------- map helpers -----------------
    def _scan_tiles(self):
        self.tiles = {}
        for x in range(self.map.width):
            for y in range(self.map.height):
                name = getattr(self.map.tiles[x][y], "tile_name", "")
                self.tiles.setdefault(name, []).append((x, y))

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

    def _choose_assembly_counter(self):
        counters = self.tiles.get("COUNTER", [])
        submits = self.tiles.get("SUBMIT", [])
        boxes = self.tiles.get("BOX", [])
        if len(counters) == 1 and boxes:
            if not submits:
                return boxes[0]
            best = None
            best_dist = 10**9
            for b in boxes:
                dist = min(max(abs(b[0] - s[0]), abs(b[1] - s[1])) for s in submits)
                if dist < best_dist:
                    best_dist = dist
                    best = b
            return best
        if not counters:
            return None
        if not submits:
            return counters[0]
        best = None
        best_dist = 10**9
        for c in counters:
            dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in submits)
            if dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _choose_prep_counter(self):
        counters = self.tiles.get("COUNTER", [])
        shops = self.tiles.get("SHOP", [])
        if not counters:
            return None
        candidates = [c for c in counters if c != self.assembly_counter] or counters
        if not shops:
            return candidates[0]
        best = None
        best_dist = 10**9
        for c in candidates:
            dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in shops)
            if dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _choose_cooker(self):
        cookers = self.tiles.get("COOKER", [])
        if not cookers:
            return None
        if self.assembly_counter:
            ax, ay = self.assembly_counter
            return self._nearest(cookers, ax, ay)
        return cookers[0]

    # ----------------- movement -----------------
    def _bfs_step(self, controller: RobotController, start, goal_fn, blocked):
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
                    if (nx, ny) in visited or (nx, ny) in blocked:
                        continue
                    if 0 <= nx < m.width and 0 <= ny < m.height and m.is_tile_walkable(nx, ny):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def _move_or_action(self, controller: RobotController, bot_id: int, tx: int, ty: int, action_fn) -> bool:
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return False
        bx, by = bot["x"], bot["y"]
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return bool(action_fn())

        blocked = set()
        for other_id in controller.get_team_bot_ids(controller.get_team()):
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other:
                blocked.add((other["x"], other["y"]))

        step = self._bfs_step(controller, (bx, by), lambda x, y: max(abs(x - tx), abs(y - ty)) <= 1, blocked)
        if step is None:
            step = self._bfs_step(controller, (bx, by), lambda x, y: max(abs(x - tx), abs(y - ty)) <= 1, set())
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            nbx, nby = bx + step[0], by + step[1]
            if max(abs(nbx - tx), abs(nby - ty)) <= 1:
                action_fn()
            return True
        return False

    def _move_then_act(self, controller: RobotController, bot_id: int, tx: int, ty: int, action_fn) -> bool:
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return False
        bx, by = bot["x"], bot["y"]
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return bool(action_fn())

        blocked = set()
        for other_id in controller.get_team_bot_ids(controller.get_team()):
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other:
                blocked.add((other["x"], other["y"]))

        step = self._bfs_step(controller, (bx, by), lambda x, y: max(abs(x - tx), abs(y - ty)) <= 1, blocked)
        if step is None:
            step = self._bfs_step(controller, (bx, by), lambda x, y: max(abs(x - tx), abs(y - ty)) <= 1, set())
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
        return False

    # ----------------- orders -----------------
    def _active_orders(self, controller: RobotController):
        orders = controller.get_orders(controller.get_team())
        return [o for o in orders if o.get("is_active")]

    def _order_foods(self, order) -> List[FoodType]:
        out: List[FoodType] = []
        for name in order.get("required", []):
            try:
                out.append(FoodType[name.upper()])
            except Exception:
                continue
        return out

    def _choose_order(self, controller: RobotController):
        orders = self._active_orders(controller)
        if not orders:
            return None
        current_turn = controller.get_turn()
        team_money = controller.get_team_money(controller.get_team())
        scored = []
        for o in orders:
            foods = self._order_foods(o)
            cost = sum(int(getattr(f, "buy_cost", 0)) for f in foods) + int(ShopCosts.PLATE.buy_cost)
            reward = int(o.get("reward", 0))
            penalty = int(o.get("penalty", 0))
            created = int(o.get("created_turn", 0))
            expires = int(o.get("expires_turn", 0))
            remaining_turns = expires - current_turn
            has_cooking = any(f.can_cook for f in foods)
            min_turns_needed = 67 if has_cooking else 40
            if remaining_turns < min_turns_needed:
                continue

            num_ingredients = len(foods)
            cooking_count = sum(1 for f in foods if f.can_cook)
            estimated_turns = num_ingredients * 20 + cooking_count * 30
            if remaining_turns < estimated_turns * 1.3:
                continue

            effort = 0
            for f in foods:
                effort += 1
                if f.can_chop:
                    effort += 2
                if f.can_cook:
                    effort += 3
            
            total_value = reward + penalty
            valuable = cost < total_value
            adjusted_reward = total_value - effort * 10
            score = (1 if valuable else 0, adjusted_reward, total_value, -expires - created, -len(foods))
            scored.append((score, o))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[0][3], item[0][4]))
        return scored[0][1]

    # ----------------- plate helpers -----------------
    def _plate_food_counter(self, plate_obj) -> Counter:
        counter = Counter()
        if plate_obj is None:
            return counter
        if isinstance(plate_obj, Plate):
            for f in plate_obj.food:
                if isinstance(f, Food):
                    counter[f.food_name] += 1
            return counter
        if isinstance(plate_obj, dict):
            for f in plate_obj.get("food", []) or []:
                name = f.get("food_name")
                if name:
                    counter[name] += 1
        return counter

    def _get_plate_location(self, controller: RobotController, bot_ids: List[int]):
        if self.assembly_counter is not None:
            ax, ay = self.assembly_counter
            tile = controller.get_tile(controller.get_team(), ax, ay)
            if tile and isinstance(getattr(tile, "item", None), Plate):
                return ("counter", (ax, ay), tile.item)
        for bid in bot_ids:
            bot = controller.get_bot_state(bid)
            holding = bot.get("holding") if bot else None
            if holding and holding.get("type") == "Plate":
                return ("bot", bid, holding)
        return None

    def _ensure_plate_on_counter(self, controller: RobotController, bot_id: int) -> bool:
        if self.assembly_counter is None:
            return False
        ax, ay = self.assembly_counter
        tile = controller.get_tile(controller.get_team(), ax, ay)
        if tile and isinstance(getattr(tile, "item", None), Plate):
            return True

        bot = controller.get_bot_state(bot_id)
        holding = bot.get("holding") if bot else None
        if holding and holding.get("type") == "Plate":
            return self._move_or_action(
                controller,
                bot_id,
                ax,
                ay,
                lambda: controller.place(bot_id, ax, ay),
            )

        if holding is not None:
            return False

        sinktables = self.tiles.get("SINKTABLE", [])
        if sinktables:
            sx, sy = self._nearest(sinktables, bot["x"], bot["y"])
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, "num_clean_plates", 0) > 0:
                def take_plate():
                    return controller.take_clean_plate(bot_id, sx, sy)
                if self._move_or_action(controller, bot_id, sx, sy, take_plate):
                    return False

        shop = self._nearest(self.tiles.get("SHOP", []), bot["x"], bot["y"]) if bot else None
        if shop and self._move_or_action(
            controller,
            bot_id,
            shop[0],
            shop[1],
            lambda: controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
            if controller.can_buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
            else False,
        ):
            return False
        return False

    def _submit_plate(self, controller: RobotController, bot_id: int):
        if self.assembly_counter is None:
            return False
        submit = self._nearest(self.tiles.get("SUBMIT", []), self.assembly_counter[0], self.assembly_counter[1])
        if not submit:
            return False

        bot = controller.get_bot_state(bot_id)
        holding = bot.get("holding") if bot else None
        if holding and holding.get("type") == "Plate" and not holding.get("dirty"):
            return self._move_or_action(
                controller,
                bot_id,
                submit[0],
                submit[1],
                lambda: controller.submit(bot_id, submit[0], submit[1]),
            )

        ax, ay = self.assembly_counter
        return self._move_or_action(
            controller,
            bot_id,
            ax,
            ay,
            lambda: controller.pickup(bot_id, ax, ay),
        )

    def _clear_hands(self, controller: RobotController, bot_id: int) -> bool:
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        holding = bot.get("holding")
        if holding is None:
            return False

        bx, by = bot["x"], bot["y"]
        if holding.get("type") == "Plate":
            target = self._nearest(self.tiles.get("COUNTER", []), bx, by)
            if target is None:
                target = self._nearest(self.tiles.get("BOX", []), bx, by)
            if target:
                tx, ty = target
                return self._move_or_action(controller, bot_id, tx, ty, lambda: controller.place(bot_id, tx, ty))
            return False

        trash = self._nearest(self.tiles.get("TRASH", []), bx, by)
        if trash:
            tx, ty = trash
            return self._move_or_action(controller, bot_id, tx, ty, lambda: controller.trash(bot_id, tx, ty))
        return False

    # ----------------- item helpers -----------------
    def _food_needed(self, missing: Counter, name: str) -> bool:
        return missing.get(name, 0) > 0

    def _add_to_plate(self, controller: RobotController, bot_id: int) -> bool:
        if self.assembly_counter is None:
            return False
        ax, ay = self.assembly_counter
        return self._move_or_action(
            controller,
            bot_id,
            ax,
            ay,
            lambda: controller.add_food_to_plate(bot_id, ax, ay),
        )

    def _find_empty_counter(self, controller: RobotController, bot_id: int) -> Optional[Tuple[int, int]]:
        counters = self.tiles.get("COUNTER", [])
        if not counters:
            return None
        bot = controller.get_bot_state(bot_id)
        if bot:
            bx, by = bot["x"], bot["y"]
        else:
            bx, by = 0, 0
        best = None
        best_dist = 10**9
        for x, y in counters:
            if self.assembly_counter == (x, y):
                tile = controller.get_tile(controller.get_team(), x, y)
                if tile and isinstance(getattr(tile, "item", None), Plate):
                    continue
            tile = controller.get_tile(controller.get_team(), x, y)
            if tile and getattr(tile, "item", None) is None:
                dist = max(abs(bx - x), abs(by - y))
                if dist < best_dist:
                    best_dist = dist
                    best = (x, y)
        return best

    def _ensure_pan(self, controller: RobotController, bot_id: int) -> bool:
        if self.cooker_tile is None:
            return False
        cx, cy = self.cooker_tile
        tile = controller.get_tile(controller.get_team(), cx, cy)
        if tile and isinstance(getattr(tile, "item", None), Pan):
            return True

        bot = controller.get_bot_state(bot_id)
        holding = bot.get("holding") if bot else None
        if holding is not None and holding.get("type") != "Pan":
            counter = self._find_empty_counter(controller, bot_id)
            if counter is not None:
                ex, ey = counter
                self._move_or_action(controller, bot_id, ex, ey, lambda: controller.place(bot_id, ex, ey))
                return False
        if holding and holding.get("type") == "Pan":
            self._move_or_action(
                controller,
                bot_id,
                cx,
                cy,
                lambda: controller.place(bot_id, cx, cy),
            )
            return False

        if holding is None:
            shop = self._nearest(self.tiles.get("SHOP", []), bot["x"], bot["y"]) if bot else None
            if shop:
                self._move_or_action(
                    controller,
                    bot_id,
                    shop[0],
                    shop[1],
                    lambda: controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    if controller.can_buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    else False,
                )
                return False
        return False

    def _take_cooked_food(self, controller: RobotController, bot_id: int, missing: Counter) -> bool:
        if self.cooker_tile is None:
            return False
        cx, cy = self.cooker_tile
        tile = controller.get_tile(controller.get_team(), cx, cy)
        pan = getattr(tile, "item", None)
        if not isinstance(pan, Pan):
            return False
        food = pan.food if isinstance(pan.food, Food) else None
        if food and food.cooked_stage >= 2:
            return self._move_or_action(controller, bot_id, cx, cy, lambda: controller.take_from_pan(bot_id, cx, cy))
        if food and food.cooked_stage == 1 and self._food_needed(missing, food.food_name):
            return self._move_or_action(controller, bot_id, cx, cy, lambda: controller.take_from_pan(bot_id, cx, cy))
        return False

    def _find_food_on_counters(self, controller: RobotController, missing: Counter):
        found = []
        for (x, y) in self.tiles.get("COUNTER", []):
            tile = controller.get_tile(controller.get_team(), x, y)
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and self._food_needed(missing, item.food_name):
                found.append((x, y, item))
        return found

    # ----------------- role logic -----------------
    def _assign_roles(self, controller: RobotController, bot_ids: List[int]):
        if self.role_by_bot and set(self.role_by_bot.keys()) == set(bot_ids):
            return

        self.role_by_bot = {}
        positions = {bid: controller.get_bot_state(bid) for bid in bot_ids}

        cooker = self.cooker_tile
        submit = self._nearest(self.tiles.get("SUBMIT", []), *(self.assembly_counter or (0, 0)))
        shop = self._nearest(self.tiles.get("SHOP", []), *(self.assembly_counter or (0, 0)))

        unassigned = set(bot_ids)

        def pick_closest(target):
            if not target or not unassigned:
                return None
            tx, ty = target
            best = None
            best_dist = 10**9
            for bid in list(unassigned):
                pos = positions[bid]
                if not pos:
                    continue
                dist = max(abs(pos["x"] - tx), abs(pos["y"] - ty))
                if dist < best_dist:
                    best_dist = dist
                    best = bid
            if best is not None:
                unassigned.remove(best)
            return best

        plate_bot = pick_closest(submit)
        cook_bot = pick_closest(cooker)
        prep_bot = pick_closest(shop)

        if plate_bot is not None:
            self.role_by_bot[plate_bot] = "plate"
        if cook_bot is not None and cook_bot not in self.role_by_bot:
            self.role_by_bot[cook_bot] = "cook"
        if prep_bot is not None and prep_bot not in self.role_by_bot:
            self.role_by_bot[prep_bot] = "prep"
        for bid in unassigned:
            self.role_by_bot[bid] = "runner"

    def _pick_food_for_role(self, missing: Counter, role: str, strict_roles: bool) -> Optional[FoodType]:
        if not missing:
            return None
        candidates = []
        for name, count in missing.items():
            if count <= 0:
                continue
            try:
                ft = FoodType[name]
            except Exception:
                continue
            if strict_roles:
                if ft.can_cook and role not in {"cook"}:
                    continue
                if ft.can_chop and not ft.can_cook and role not in {"prep"}:
                    continue
                if (not ft.can_chop and not ft.can_cook) and role not in {"plate", "runner", "prep", "cook"}:
                    continue
            steps = 1 + (1 if ft.can_chop else 0) + (1 if ft.can_cook else 0)
            bonus = 0
            if role == "cook" and ft.can_cook:
                bonus += 3
            if role == "prep" and ft.can_chop:
                bonus += 3
            if role == "plate" and (not ft.can_chop and not ft.can_cook):
                bonus += 3
            if role == "runner":
                bonus += 1
            candidates.append((bonus, -steps, ft.buy_cost, ft))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    def _handle_holding_food(self, controller: RobotController, bot_id: int, food_type: FoodType, missing: Counter) -> bool:
        bot = controller.get_bot_state(bot_id)
        holding = bot.get("holding") if bot else None
        if not holding or holding.get("type") != "Food":
            return False

        if int(holding.get("cooked_stage", 0)) >= 2:
            trash = self._nearest(self.tiles.get("TRASH", []), bot["x"], bot["y"]) if bot else None
            if trash:
                return self._move_or_action(controller, bot_id, trash[0], trash[1], lambda: controller.trash(bot_id, trash[0], trash[1]))
            return False

        if not self._food_needed(missing, food_type.food_name):
            counter = self._find_empty_counter(controller, bot_id)
            if counter is not None:
                cx, cy = counter
                return self._move_or_action(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy))
            trash = self._nearest(self.tiles.get("TRASH", []), bot["x"], bot["y"]) if bot else None
            if trash:
                return self._move_or_action(controller, bot_id, trash[0], trash[1], lambda: controller.trash(bot_id, trash[0], trash[1]))
            return False

        if food_type.can_chop and not holding.get("chopped", False):
            counter = self._find_empty_counter(controller, bot_id)
            if counter is None:
                return False
            cx, cy = counter
            return self._move_or_action(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy))

        if food_type.can_cook and int(holding.get("cooked_stage", 0)) == 0:
            if not self._ensure_pan(controller, bot_id):
                return True
            if self.cooker_tile is None:
                return False
            cx, cy = self.cooker_tile
            tile = controller.get_tile(controller.get_team(), cx, cy)
            pan = getattr(tile, "item", None)
            if not isinstance(pan, Pan) or pan.food is not None:
                return False
            return self._move_or_action(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy))

        return self._add_to_plate(controller, bot_id)

    def _handle_plate_holding(self, controller: RobotController, bot_id: int, missing: Counter, role: str, bad_plate: bool) -> bool:
        bot = controller.get_bot_state(bot_id)
        holding = bot.get("holding") if bot else None
        if not holding or holding.get("type") != "Plate":
            return False
        if bad_plate:
            trash = self._nearest(self.tiles.get("TRASH", []), bot["x"], bot["y"]) if bot else None
            if trash:
                return self._move_or_action(controller, bot_id, trash[0], trash[1], lambda: controller.trash(bot_id, trash[0], trash[1]))
            return False
        if holding.get("dirty"):
            trash = self._nearest(self.tiles.get("TRASH", []), bot["x"], bot["y"]) if bot else None
            if trash:
                return self._move_or_action(controller, bot_id, trash[0], trash[1], lambda: controller.trash(bot_id, trash[0], trash[1]))
            return False

        if not missing:
            return self._submit_plate(controller, bot_id)

        if self.assembly_counter is not None:
            ax, ay = self.assembly_counter
            tile = controller.get_tile(controller.get_team(), ax, ay)
            if tile and getattr(tile, "item", None) is None and role in {"plate", "runner"}:
                return self._move_or_action(controller, bot_id, ax, ay, lambda: controller.place(bot_id, ax, ay))
        return False

    def _handle_idle(self, controller: RobotController, bot_id: int, missing: Counter, role: str, target_food: Optional[FoodType], bad_plate: bool) -> bool:
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return False

        if bad_plate:
            return False

        if not missing:
            return self._submit_plate(controller, bot_id)

        if role == "cook" and any(
            FoodType[name].can_cook for name in missing.keys() if name in FoodType.__members__
        ):
            if not self._ensure_pan(controller, bot_id):
                return True

        if role == "plate" and self.assembly_counter is not None:
            if not self._ensure_plate_on_counter(controller, bot_id):
                return True

        if self._take_cooked_food(controller, bot_id, missing):
            return True

        foods_on_counters = self._find_food_on_counters(controller, missing)
        for x, y, item in foods_on_counters:
            if item.can_chop and not item.chopped:
                if self._move_or_action(controller, bot_id, x, y, lambda: controller.chop(bot_id, x, y)):
                    return True
        if foods_on_counters:
            x, y, _ = foods_on_counters[0]
            return self._move_or_action(controller, bot_id, x, y, lambda: controller.pickup(bot_id, x, y))

        if target_food is None:
            return False

        shop = self._nearest(self.tiles.get("SHOP", []), bot["x"], bot["y"]) if bot else None
        if not shop:
            return False

        return self._move_or_action(
            controller,
            bot_id,
            shop[0],
            shop[1],
            lambda: controller.buy(bot_id, target_food, shop[0], shop[1])
            if controller.can_buy(bot_id, target_food, shop[0], shop[1])
            else False,
        )

    def _drive_bot(self, controller: RobotController, bot_id: int, missing: Counter, role: str, target_food: Optional[FoodType], bad_plate: bool):
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return

        holding = bot.get("holding")
        if self._handle_plate_holding(controller, bot_id, missing, role, bad_plate):
            return

        if holding and holding.get("type") == "Food":
            try:
                food_type = FoodType[holding.get("food_name")]
            except Exception:
                return
            if self._handle_holding_food(controller, bot_id, food_type, missing):
                return

        if holding is not None:
            return

        self._handle_idle(controller, bot_id, missing, role, target_food, bad_plate)

    # ----------------- single-bot plan -----------------
    def _single_build_plan(self, order) -> List[Dict]:
        plan: List[Dict] = []
        plan.append({"type": "get_plate"})
        plan.append({"type": "place_plate"})
        for name in order.get("required", []):
            try:
                food = FoodType[name.upper()]
            except Exception:
                continue
            plan.append({"type": "buy_food", "food": food})
            if food.can_chop:
                plan.append({"type": "chop_food", "food": food})
            if food.can_cook:
                plan.append({"type": "cook_food", "food": food})
            plan.append({"type": "add_to_plate"})
        plan.append({"type": "pickup_plate"})
        plan.append({"type": "submit"})
        plan.append({"type": "wash_until_clean"})
        return plan

    def _single_get_targets(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        bot = controller.get_bot_state(bot_ids[0]) if bot_ids else None
        bx, by = (bot["x"], bot["y"]) if bot else (0, 0)
        if self.assembly_counter is None:
            self.assembly_counter = self._choose_assembly_counter()
        if self.assembly_counter is not None:
            self.assembly_counter = self._choose_accessible(controller, [self.assembly_counter], (bx, by)) or self.assembly_counter
        if self.prep_counter is None:
            self.prep_counter = self._choose_prep_counter()
        if self.prep_counter is not None:
            self.prep_counter = self._choose_accessible(controller, [self.prep_counter], (bx, by)) or self.prep_counter
        if self.cooker_tile is None:
            self.cooker_tile = self._choose_cooker()
        if self.cooker_tile is not None:
            self.cooker_tile = self._choose_accessible(controller, [self.cooker_tile], (bx, by)) or self.cooker_tile
        anchor = self.assembly_counter or (bx, by)
        sink = self._nearest(self.tiles.get("SINK", []), *anchor)
        sinktable = self._nearest(self.tiles.get("SINKTABLE", []), *anchor)
        shop = self._nearest(self.tiles.get("SHOP", []), *anchor)
        submit = self._nearest(self.tiles.get("SUBMIT", []), *anchor)
        return sink, sinktable, shop, submit

    def _single_step(self, controller: RobotController, bot_id: int, task: Dict) -> bool:
        bot = controller.get_bot_state(bot_id)
        if bot is None:
            return False
        holding = bot.get("holding")
        sink, sinktable, shop, submit = self._single_get_targets(controller)
        ax, ay = self.assembly_counter if self.assembly_counter else (None, None)

        ttype = task.get("type")

        if ttype == "get_plate":
            if holding and holding.get("type") == "Plate":
                return True
            if sinktable:
                sx, sy = sinktable
                tile = controller.get_tile(controller.get_team(), sx, sy)
                if tile and getattr(tile, "num_clean_plates", 0) > 0:
                    return self._move_then_act(controller, bot_id, sx, sy, lambda: controller.take_clean_plate(bot_id, sx, sy))
            if shop:
                sx, sy = shop
                return self._move_then_act(
                    controller,
                    bot_id,
                    sx,
                    sy,
                    lambda: controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
                    if controller.can_buy(bot_id, ShopCosts.PLATE, sx, sy)
                    else False,
                )
            return False

        if ttype == "place_plate":
            if ax is None:
                return False
            tile = controller.get_tile(controller.get_team(), ax, ay)
            if tile and isinstance(getattr(tile, "item", None), Plate):
                return True
            if holding and holding.get("type") == "Plate":
                return self._move_then_act(controller, bot_id, ax, ay, lambda: controller.place(bot_id, ax, ay))
            return False

        if ttype == "buy_food":
            food = task.get("food")
            if holding and holding.get("type") == "Food" and holding.get("food_name") == food.food_name:
                return True
            if shop:
                sx, sy = shop
                return self._move_then_act(
                    controller,
                    bot_id,
                    sx,
                    sy,
                    lambda: controller.buy(bot_id, food, sx, sy)
                    if controller.can_buy(bot_id, food, sx, sy)
                    else False,
                )
            return False

        if ttype == "chop_food":
            if holding and holding.get("type") == "Food" and holding.get("chopped"):
                return True
            counter = task.get("counter")
            if counter is None:
                counter = self._find_empty_counter(controller, bot_id)
                if counter is None:
                    return False
                task["counter"] = counter
            cx, cy = counter
            stage = task.get("stage", "place")
            if stage == "place":
                tile = controller.get_tile(controller.get_team(), cx, cy)
                item = getattr(tile, "item", None) if tile else None
                if holding is None and isinstance(item, Food):
                    task["stage"] = "chop" if not item.chopped else "pickup"
                    return False
                if holding is not None and holding.get("type") == "Food":
                    if self._move_then_act(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy)):
                        task["stage"] = "chop"
                return False
            if stage == "chop":
                if self._move_then_act(controller, bot_id, cx, cy, lambda: controller.chop(bot_id, cx, cy)):
                    task["stage"] = "pickup"
                return False
            if stage == "pickup":
                return self._move_then_act(controller, bot_id, cx, cy, lambda: controller.pickup(bot_id, cx, cy))
            return False

        if ttype == "cook_food":
            if self.cooker_tile is None:
                return False
            cx, cy = self.cooker_tile
            stage = task.get("stage", "place")
            if stage == "place":
                if holding is None or holding.get("type") != "Food":
                    return False
                if holding.get("cooked_stage", 0) >= 1:
                    return True
                if not self._ensure_pan(controller, bot_id):
                    return False
                tile = controller.get_tile(controller.get_team(), cx, cy)
                pan = getattr(tile, "item", None)
                if not isinstance(pan, Pan) or pan.food is not None:
                    return False
                if self._move_then_act(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy)):
                    task["stage"] = "wait"
                return False
            if stage == "wait":
                tile = controller.get_tile(controller.get_team(), cx, cy)
                pan = getattr(tile, "item", None)
                food = pan.food if isinstance(pan, Pan) else None
                if isinstance(food, Food) and food.cooked_stage == 1:
                    task["stage"] = "pickup"
                return False
            if stage == "pickup":
                return self._move_then_act(controller, bot_id, cx, cy, lambda: controller.take_from_pan(bot_id, cx, cy))
            return False

        if ttype == "add_to_plate":
            if holding and holding.get("type") == "Food":
                if self.assembly_counter is None:
                    return False
                ax, ay = self.assembly_counter
                return self._move_then_act(controller, bot_id, ax, ay, lambda: controller.add_food_to_plate(bot_id, ax, ay))
            if holding and holding.get("type") == "Plate":
                if ax is None:
                    return False
                tile = controller.get_tile(controller.get_team(), ax, ay)
                if tile and getattr(tile, "item", None) is None:
                    return self._move_then_act(controller, bot_id, ax, ay, lambda: controller.place(bot_id, ax, ay))
            return False

        if ttype == "pickup_plate":
            if holding and holding.get("type") == "Plate":
                return True
            if ax is None:
                return False
            return self._move_then_act(controller, bot_id, ax, ay, lambda: controller.pickup(bot_id, ax, ay))

        if ttype == "submit":
            if holding and holding.get("type") == "Plate":
                if submit:
                    sx, sy = submit
                    return self._move_then_act(controller, bot_id, sx, sy, lambda: controller.submit(bot_id, sx, sy))
            return False

        if ttype == "wash_until_clean":
            if sinktable:
                sx, sy = sinktable
                tile = controller.get_tile(controller.get_team(), sx, sy)
                if tile and getattr(tile, "num_clean_plates", 0) > 0:
                    return True
            if sink:
                sx, sy = sink
                return self._move_then_act(controller, bot_id, sx, sy, lambda: controller.wash_sink(bot_id, sx, sy))
            return False

        return False

    # ----------------- main entry -----------------
    def play_turn(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return
        bot0 = controller.get_bot_state(bot_ids[0])
        bot0_pos = (bot0["x"], bot0["y"]) if bot0 else (0, 0)
        self._refresh_from_controller(controller, bot0_pos)
        active_orders = self._active_orders(controller)
        if len(bot_ids) > 1 and len(active_orders) == 1:
            req = {r.upper() for r in active_orders[0].get("required", [])}
            if req == {"NOODLES", "MEAT"}:
                bot_id = bot_ids[0]
                if not self.single_plan or self.single_plan_idx >= len(self.single_plan):
                    if active_orders[0].get("order_id") not in self.single_processed_orders:
                        self.single_processed_orders.add(active_orders[0].get("order_id"))
                        self.single_plan = self._single_build_plan(active_orders[0])
                        self.single_plan_idx = 0
                if self.single_plan:
                    task = self.single_plan[self.single_plan_idx]
                    if self._single_step(controller, bot_id, task):
                        self.single_plan_idx += 1
                return
        if len(bot_ids) == 1:
            bot_id = bot_ids[0]
            if not self.single_plan or self.single_plan_idx >= len(self.single_plan):
                if self._clear_hands(controller, bot_id):
                    return
            if not self.single_plan or self.single_plan_idx >= len(self.single_plan):
                next_order = None
                for o in active_orders:
                    if o.get("order_id") not in self.single_processed_orders:
                        next_order = o
                        break
                if next_order is None:
                    return
                self.single_processed_orders.add(next_order.get("order_id"))
                self.single_plan = self._single_build_plan(next_order)
                self.single_plan_idx = 0

            task = self.single_plan[self.single_plan_idx]
            if self._single_step(controller, bot_id, task):
                self.single_plan_idx += 1
            return

        if not self.turn_initialized:
            preferred_assembly = self._choose_assembly_counter()
            if preferred_assembly is not None:
                self.assembly_counter = self._choose_accessible(controller, [preferred_assembly], bot0_pos) or preferred_assembly
            else:
                self.assembly_counter = self._choose_accessible(
                    controller, self.tiles.get("COUNTER", []) + self.tiles.get("BOX", []), bot0_pos
                )
            self.prep_counter = self._choose_accessible(controller, self.tiles.get("COUNTER", []), bot0_pos)
            self.cooker_tile = self._choose_accessible(controller, self.tiles.get("COOKER", []), bot0_pos)
            self.turn_initialized = True
        else:
            if self.assembly_counter is not None:
                if not self._target_accessible(controller, self.assembly_counter, self._reachable_walkable(controller, bot0_pos)):
                    self.assembly_counter = self._choose_accessible(
                        controller, self.tiles.get("COUNTER", []) + self.tiles.get("BOX", []), bot0_pos
                    )
            if self.prep_counter is not None:
                if not self._target_accessible(controller, self.prep_counter, self._reachable_walkable(controller, bot0_pos)):
                    self.prep_counter = self._choose_accessible(controller, self.tiles.get("COUNTER", []), bot0_pos)
            if self.cooker_tile is not None:
                if not self._target_accessible(controller, self.cooker_tile, self._reachable_walkable(controller, bot0_pos)):
                    self.cooker_tile = self._choose_accessible(controller, self.tiles.get("COOKER", []), bot0_pos)

        self._assign_roles(controller, bot_ids)

        active_orders = self._active_orders(controller)
        if self.current_order_id is not None:
            order = next((o for o in active_orders if o.get("order_id") == self.current_order_id), None)
            if order is None:
                self.current_order_id = None
        if self.current_order_id is None:
            order = self._choose_order(controller)
            if order is not None:
                self.current_order_id = order.get("order_id")

        order = next((o for o in active_orders if o.get("order_id") == self.current_order_id), None)
        if order is None or self.assembly_counter is None:
            return

        required = Counter([f.food_name for f in self._order_foods(order)])
        plate_info = self._get_plate_location(controller, bot_ids)
        plate_counter = Counter()
        if plate_info is not None:
            plate_counter = self._plate_food_counter(plate_info[2])
        extra = plate_counter - required
        bad_plate = bool(extra)
        missing = required - plate_counter

        # prioritize plate bot first to keep the plate in place
        role_order = ["plate", "cook", "prep", "runner"]
        bots_by_role = sorted(bot_ids, key=lambda bid: role_order.index(self.role_by_bot.get(bid, "runner")))

        if bad_plate and plate_info is not None and plate_info[0] == "counter":
            plate_bot = next((bid for bid in bots_by_role if self.role_by_bot.get(bid) == "plate"), bots_by_role[0])
            ax, ay = plate_info[1]
            self._move_or_action(controller, plate_bot, ax, ay, lambda: controller.pickup(plate_bot, ax, ay))
            return

        plan_missing = missing.copy()
        strict_roles = len(bot_ids) > 1
        assignments: Dict[int, Optional[FoodType]] = {}
        for bid in bots_by_role:
            role = self.role_by_bot.get(bid, "runner")
            target_food = self._pick_food_for_role(plan_missing, role, strict_roles)
            if target_food is not None:
                plan_missing[target_food.food_name] -= 1
                if plan_missing[target_food.food_name] <= 0:
                    del plan_missing[target_food.food_name]
            assignments[bid] = target_food

        for bid in bots_by_role:
            role = self.role_by_bot.get(bid, "runner")
            self._drive_bot(controller, bid, missing, role, assignments.get(bid), bad_plate)
