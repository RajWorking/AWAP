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
        self.completed_orders = set()  # Track successfully completed orders

        # Per-bot independent recipe tracking
        self.bot_orders: Dict[int, Optional[int]] = {}  # bot_id -> order_id
        self.bot_plans: Dict[int, List[Dict]] = {}  # bot_id -> plan
        self.bot_plan_idx: Dict[int, int] = {}  # bot_id -> current plan step
        self.bot_assembly_counters: Dict[int, Optional[Tuple[int, int]]] = {}  # bot_id -> assembly counter
        self.bot_cooker_tiles: Dict[int, Optional[Tuple[int, int]]] = {}  # bot_id -> cooker tile
        self.bot_prep_counters: Dict[int, Optional[Tuple[int, int]]] = {}  # bot_id -> prep counter

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
        shops = self.tiles.get("SHOP", [])
        cookers = self.tiles.get("COOKER", [])
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
        best_score = 10**9
        for c in counters:
            submit_dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in submits) if submits else 0
            shop_dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in shops) if shops else 0
            cooker_dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in cookers) if cookers else 0
            score = submit_dist * 2 + shop_dist + cooker_dist
            if score < best_score:
                best_score = score
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
            result = action_fn()
            return bool(result)

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
            # Check if we're now adjacent after the move
            nbx, nby = bx + step[0], by + step[1]
            if max(abs(nbx - tx), abs(nby - ty)) <= 1:
                result = action_fn()
                return bool(result)
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

    def _order_signature(self, order) -> Counter:
        sig = Counter()
        for ft in self._order_foods(order):
            sig[(ft.food_name, ft.can_chop, 1 if ft.can_cook else 0)] += 1
        return sig

    def _plate_signature(self, plate_obj) -> Counter:
        sig = Counter()
        if plate_obj is None:
            return sig
        if isinstance(plate_obj, Plate):
            for f in plate_obj.food:
                if isinstance(f, Food):
                    sig[(f.food_name, bool(f.chopped), int(f.cooked_stage))] += 1
            return sig
        if isinstance(plate_obj, dict):
            for f in plate_obj.get("food", []) or []:
                name = f.get("food_name")
                if name:
                    sig[(name, bool(f.get("chopped", False)), int(f.get("cooked_stage", 0)))] += 1
        return sig

    def _find_matching_order_id(self, controller: RobotController, plate_obj) -> Optional[int]:
        if plate_obj is None:
            return None
        plate_sig = self._plate_signature(plate_obj)
        if not plate_sig:
            return None
        for o in self._active_orders(controller):
            if plate_sig == self._order_signature(o):
                return o.get("order_id")
        return None

    def _choose_order(self, controller: RobotController, bot_count: int):
        orders = self._active_orders(controller)
        # Filter out already completed orders
        orders = [o for o in orders if o.get("order_id") not in self.completed_orders]
        if not orders:
            return None
        current_turn = controller.get_turn()
        team_money = controller.get_team_money(controller.get_team())
        bot_count = max(1, bot_count)

        candidates = []
        for o in orders:
            foods = self._order_foods(o)
            if not foods:
                continue

            # Calculate costs
            cost = sum(int(getattr(f, "buy_cost", 0)) for f in foods) + int(ShopCosts.PLATE.buy_cost)
            reward = int(o.get("reward", 0))
            penalty = int(o.get("penalty", 0))
            total_value = reward + penalty

            # PROFITABILITY CHECK: Skip unprofitable orders
            if cost >= total_value:
                continue

            profit = total_value - cost

            expires = int(o.get("expires_turn", 0))
            remaining_turns = expires - current_turn
            if remaining_turns <= 0:
                continue

            num_ingredients = len(foods)
            cooking_count = sum(1 for f in foods if f.can_cook)
            chop_count = sum(1 for f in foods if f.can_chop)

            # OPTIMIZED TIME ESTIMATION for independent bots (no washing step)
            base_turns = 10  # Base overhead (reduced since no washing)
            ingredient_turns = num_ingredients * 4  # Time per ingredient
            chopping_turns = chop_count * 3  # Chopping time
            cooking_turns = cooking_count * 20  # Cooking time per item

            estimated_turns = base_turns + ingredient_turns + chopping_turns + cooking_turns

            # More aggressive BUFFER (we're faster now)
            game_turns_remaining = 500 - current_turn
            if game_turns_remaining < 100:
                # Late game: ensure completion
                buffer_multiplier = 1.15
            elif game_turns_remaining < 250:
                # Mid game: aggressive
                buffer_multiplier = 1.05
            else:
                # Early game: very aggressive
                buffer_multiplier = 1.02

            # Skip if not enough time
            if remaining_turns < estimated_turns * buffer_multiplier:
                continue

            # Check if we can afford it
            feasible = cost <= team_money
            if not feasible:
                continue

            # Calculate efficiency - prioritize speed AND profit
            profit_per_turn = profit / estimated_turns if estimated_turns > 0 else 0

            # Complexity penalty - simpler orders are better
            complexity = cooking_count * 2 + chop_count

            candidates.append({
                'order': o,
                'cost': cost,
                'profit': profit,
                'profit_per_turn': profit_per_turn,
                'estimated_turns': estimated_turns,
                'remaining_turns': remaining_turns,
                'expires': expires,
                'num_ingredients': num_ingredients,
                'complexity': complexity
            })

        if not candidates:
            return None

        # OPTIMIZED STRATEGY: Speed + Profit
        game_turns_remaining = 500 - current_turn

        for c in candidates:
            # Favor fast, profitable orders
            speed_bonus = 100 / max(c['estimated_turns'], 1)
            profit_bonus = c['profit'] * 0.8
            complexity_penalty = c['complexity'] * 2
            c['score'] = speed_bonus + profit_bonus + c['profit_per_turn'] * 60 - complexity_penalty

        if game_turns_remaining < 100:
            # Late game: prioritize fastest orders that fit
            candidates.sort(key=lambda c: c['estimated_turns'])
        else:
            # Early/mid game: maximize score
            candidates.sort(key=lambda c: -c['score'])

        return candidates[0]['order']

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

    def _collect_inflight_food(self, controller: RobotController, required_names: set) -> Counter:
        inflight = Counter()
        if not required_names:
            return inflight

        for (x, y) in self.tiles.get("COUNTER", []):
            tile = controller.get_tile(controller.get_team(), x, y)
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and item.food_name in required_names and item.cooked_stage < 2:
                inflight[item.food_name] += 1

        for (x, y) in self.tiles.get("COOKER", []):
            tile = controller.get_tile(controller.get_team(), x, y)
            pan = getattr(tile, "item", None)
            food = pan.food if isinstance(pan, Pan) else None
            if isinstance(food, Food) and food.food_name in required_names and food.cooked_stage < 2:
                inflight[food.food_name] += 1

        return inflight

    def _collect_holding_food(self, controller: RobotController, bot_ids: List[int], required_names: set):
        holdings = []
        if not required_names:
            return holdings
        for bid in bot_ids:
            bot = controller.get_bot_state(bid)
            if not bot:
                continue
            holding = bot.get("holding")
            if holding and holding.get("type") == "Food":
                name = holding.get("food_name")
                if name and name in required_names:
                    holdings.append(
                        {
                            "bot_id": bid,
                            "name": name,
                            "cooked_stage": int(holding.get("cooked_stage", 0)),
                            "chopped": bool(holding.get("chopped", False)),
                            "pos": (bot["x"], bot["y"]),
                        }
                    )
        return holdings

    def _select_holdings_to_keep(
        self,
        holdings: List[Dict],
        required: Counter,
        plate_counter: Counter,
        inflight_tiles: Counter,
    ):
        keep_bots = set()
        kept_counter = Counter()
        if not holdings:
            return keep_bots, kept_counter

        by_name: Dict[str, List[Dict]] = {}
        for h in holdings:
            by_name.setdefault(h["name"], []).append(h)

        ax, ay = self.assembly_counter if self.assembly_counter is not None else (None, None)

        for name, group in by_name.items():
            needed = required.get(name, 0) - plate_counter.get(name, 0) - inflight_tiles.get(name, 0)
            if needed <= 0:
                continue

            candidates = [g for g in group if g["cooked_stage"] < 2]
            if not candidates:
                continue

            def score(entry):
                cooked_stage = entry["cooked_stage"]
                cooked_score = 2 if cooked_stage == 1 else 1 if cooked_stage == 0 else 0
                chopped_score = 1 if entry["chopped"] else 0
                dist_score = 0
                if ax is not None:
                    dist_score = -max(abs(entry["pos"][0] - ax), abs(entry["pos"][1] - ay))
                return (cooked_score, chopped_score, dist_score)

            candidates.sort(key=score, reverse=True)
            for entry in candidates:
                if needed <= 0:
                    break
                keep_bots.add(entry["bot_id"])
                kept_counter[name] += 1
                needed -= 1

        return keep_bots, kept_counter

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

        counters = self.tiles.get("COUNTER", [])
        if counters and bot:
            bx, by = bot["x"], bot["y"]
            best = None
            best_dist = 10**9
            for x, y in counters:
                tile = controller.get_tile(controller.get_team(), x, y)
                item = getattr(tile, "item", None)
                if isinstance(item, Plate) and not item.dirty:
                    dist = max(abs(bx - x), abs(by - y))
                    if dist < best_dist:
                        best_dist = dist
                        best = (x, y)
            if best:
                px, py = best
                if self._move_or_action(controller, bot_id, px, py, lambda: controller.pickup(bot_id, px, py)):
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
        tile = controller.get_tile(controller.get_team(), ax, ay)
        if tile and isinstance(getattr(tile, "item", None), Plate):
            return self._move_or_action(
                controller,
                bot_id,
                ax,
                ay,
                lambda: controller.pickup(bot_id, ax, ay),
            )
        return False

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

    def _find_food_on_counters(self, controller: RobotController, missing: Counter, bot_pos: Optional[Tuple[int, int]] = None):
        found = []
        for (x, y) in self.tiles.get("COUNTER", []):
            tile = controller.get_tile(controller.get_team(), x, y)
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and self._food_needed(missing, item.food_name):
                found.append((x, y, item))
        if bot_pos is not None:
            bx, by = bot_pos
            found.sort(key=lambda item: max(abs(item[0] - bx), abs(item[1] - by)))
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
        def build_candidates(apply_roles: bool):
            items = []
            for name, count in missing.items():
                if count <= 0:
                    continue
                try:
                    ft = FoodType[name]
                except Exception:
                    continue
                if apply_roles:
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
                items.append((bonus, -steps, ft.buy_cost, ft))
            return items

        candidates = build_candidates(strict_roles)
        if not candidates and strict_roles:
            candidates = build_candidates(False)
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

        foods_on_counters = self._find_food_on_counters(controller, missing, (bot["x"], bot["y"]))
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

        # Check if we need to cook anything
        needs_pan = False
        for name in order.get("required", []):
            try:
                food = FoodType[name.upper()]
                if food.can_cook:
                    needs_pan = True
                    break
            except Exception:
                continue

        # Get pan first if needed
        if needs_pan:
            plan.append({"type": "get_pan"})

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
        # Skip washing - just get new plates next time for speed
        # plan.append({"type": "wash_until_clean"})
        return plan

    def _single_get_targets(self, controller: RobotController, bot_id: int):
        bot = controller.get_bot_state(bot_id)
        bx, by = (bot["x"], bot["y"]) if bot else (0, 0)

        # Get or assign this bot's assembly counter
        if bot_id not in self.bot_assembly_counters or self.bot_assembly_counters[bot_id] is None:
            # Try to find an unused counter close to shop and submit
            counters = self.tiles.get("COUNTER", [])
            used_counters = set(c for c in self.bot_assembly_counters.values() if c is not None)
            available = [c for c in counters if c not in used_counters]

            if not available:
                # Fall back to any counter
                available = counters

            if available:
                # Prioritize counters close to shop and submit for efficiency
                shops = self.tiles.get("SHOP", [])
                submits = self.tiles.get("SUBMIT", [])
                if shops and submits:
                    best = None
                    best_score = float('inf')
                    for c in available:
                        shop_dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in shops)
                        submit_dist = min(max(abs(c[0] - s[0]), abs(c[1] - s[1])) for s in submits)
                        score = shop_dist + submit_dist * 2  # Prioritize proximity to submit
                        if score < best_score:
                            best_score = score
                            best = c
                    if best:
                        self.bot_assembly_counters[bot_id] = best
                    else:
                        self.bot_assembly_counters[bot_id] = self._choose_accessible(controller, available, (bx, by)) or available[0]
                else:
                    self.bot_assembly_counters[bot_id] = self._choose_accessible(controller, available, (bx, by)) or available[0]
            else:
                # No counters, use box
                boxes = self.tiles.get("BOX", [])
                if boxes:
                    self.bot_assembly_counters[bot_id] = boxes[0]

        # Get or assign this bot's cooker tile
        if bot_id not in self.bot_cooker_tiles or self.bot_cooker_tiles[bot_id] is None:
            cookers = self.tiles.get("COOKER", [])
            used_cookers = set(c for c in self.bot_cooker_tiles.values() if c is not None)
            available_cookers = [c for c in cookers if c not in used_cookers]

            if not available_cookers:
                # Fall back to any cooker
                available_cookers = cookers

            if available_cookers:
                self.bot_cooker_tiles[bot_id] = self._nearest(available_cookers, bx, by)

        # Get or assign this bot's prep counter
        if bot_id not in self.bot_prep_counters or self.bot_prep_counters[bot_id] is None:
            counters = self.tiles.get("COUNTER", [])
            used_counters = set(c for c in self.bot_assembly_counters.values() if c is not None)
            used_counters.update(c for c in self.bot_prep_counters.values() if c is not None)
            available = [c for c in counters if c not in used_counters]

            if not available:
                available = counters

            if available:
                self.bot_prep_counters[bot_id] = self._nearest(available, bx, by)

        anchor = self.bot_assembly_counters.get(bot_id) or (bx, by)
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
        sink, sinktable, shop, submit = self._single_get_targets(controller, bot_id)
        ax, ay = self.bot_assembly_counters.get(bot_id) or (None, None)

        ttype = task.get("type")

        if ttype == "get_plate":
            if holding and holding.get("type") == "Plate":
                return True

            # Prioritize buying plates when we have money for speed
            team_money = controller.get_team_money(controller.get_team())
            if team_money >= ShopCosts.PLATE.buy_cost * 3 and shop:
                # We're rich, buy for speed
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

            # Otherwise try sinktable first
            if sinktable:
                sx, sy = sinktable
                tile = controller.get_tile(controller.get_team(), sx, sy)
                if tile and getattr(tile, "num_clean_plates", 0) > 0:
                    return self._move_then_act(controller, bot_id, sx, sy, lambda: controller.take_clean_plate(bot_id, sx, sy))

            # Fall back to buying
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

        if ttype == "get_pan":
            # Get pan and place on cooker
            cooker = self.bot_cooker_tiles.get(bot_id)
            if cooker is None:
                return False
            cx, cy = cooker
            tile = controller.get_tile(controller.get_team(), cx, cy)
            pan = getattr(tile, "item", None)
            if isinstance(pan, Pan):
                # Pan already on cooker
                return True
            if holding and holding.get("type") == "Pan":
                # Place pan on cooker
                return self._move_then_act(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy))
            if holding is None:
                # Buy pan
                if shop:
                    sx, sy = shop
                    return self._move_then_act(
                        controller,
                        bot_id,
                        sx,
                        sy,
                        lambda: controller.buy(bot_id, ShopCosts.PAN, sx, sy)
                        if controller.can_buy(bot_id, ShopCosts.PAN, sx, sy)
                        else False,
                    )
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
            # Use bot-specific prep counter
            counter = task.get("counter")
            if counter is None:
                counter = self.bot_prep_counters.get(bot_id)
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
            # Use bot-specific cooker
            cooker = self.bot_cooker_tiles.get(bot_id)
            if cooker is None:
                return False
            cx, cy = cooker
            stage = task.get("stage", "place")
            if stage == "place":
                if holding is None or holding.get("type") != "Food":
                    return False
                if holding.get("cooked_stage", 0) >= 1:
                    return True
                # Check if pan exists
                tile = controller.get_tile(controller.get_team(), cx, cy)
                pan = getattr(tile, "item", None)
                if not isinstance(pan, Pan):
                    # No pan - this shouldn't happen if get_pan step worked
                    return False
                if pan.food is not None:
                    # Pan already has food, need to wait for it or clear it
                    # For now, just wait
                    return False
                # Place food in pan
                if self._move_then_act(controller, bot_id, cx, cy, lambda: controller.place(bot_id, cx, cy)):
                    task["stage"] = "wait"
                return False
            if stage == "wait":
                tile = controller.get_tile(controller.get_team(), cx, cy)
                pan = getattr(tile, "item", None)
                food = pan.food if isinstance(pan, Pan) else None
                if isinstance(food, Food):
                    if food.cooked_stage == 1:
                        # Perfect! Pick it up now
                        task["stage"] = "pickup"
                    elif food.cooked_stage >= 2:
                        # Burned! Still need to pick up to avoid blocking
                        task["stage"] = "pickup"
                return False
            if stage == "pickup":
                return self._move_then_act(controller, bot_id, cx, cy, lambda: controller.take_from_pan(bot_id, cx, cy))
            return False

        if ttype == "add_to_plate":
            if holding and holding.get("type") == "Food":
                if ax is None:
                    return False
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
            # Skip washing if we have money - just buy new plates for speed
            if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost * 2:
                return True
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

    def _cleanup_workspace(self, controller: RobotController, bot_ids: List[int]) -> bool:
        """Clear hands and workspace after order expires or fails. Returns True if still cleaning."""
        # Clear hands for all bots holding items
        for bot_id in bot_ids:
            bot = controller.get_bot_state(bot_id)
            if not bot:
                continue
            holding = bot.get("holding")
            if holding is None:
                continue

            bx, by = bot["x"], bot["y"]
            holding_type = holding.get("type") if isinstance(holding, dict) else "unknown"

            # Special case: clean empty plates need to be placed, not trashed
            if holding_type == "Plate":
                food_on_plate = holding.get("food", []) if isinstance(holding, dict) else []
                is_dirty = holding.get("dirty", False) if isinstance(holding, dict) else False

                if not food_on_plate and not is_dirty:
                    # Place clean empty plate
                    counters = self.tiles.get("COUNTER", [])
                    if counters:
                        target = self._nearest(counters, bx, by)
                        if target:
                            tile = controller.get_tile(controller.get_team(), target[0], target[1])
                            if tile and getattr(tile, "item", None) is None:
                                self._move_or_action(controller, bot_id, target[0], target[1],
                                                    lambda: controller.place(bot_id, target[0], target[1]))
                                return True
                    # Try box if counter full
                    boxes = self.tiles.get("BOX", [])
                    if boxes:
                        target = self._nearest(boxes, bx, by)
                        if target:
                            self._move_or_action(controller, bot_id, target[0], target[1],
                                                lambda: controller.place(bot_id, target[0], target[1]))
                            return True
                    continue

            # Trash everything else (food, dirty plates, pans)
            trash = self._nearest(self.tiles.get("TRASH", []), bx, by)
            if trash:
                self._move_or_action(controller, bot_id, trash[0], trash[1],
                                    lambda: controller.trash(bot_id, trash[0], trash[1]))
                return True

        return False  # All hands cleared

    # ----------------- main entry -----------------
    def play_turn(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return
        bot0 = controller.get_bot_state(bot_ids[0])
        bot0_pos = (bot0["x"], bot0["y"]) if bot0 else (0, 0)
        self._refresh_from_controller(controller, bot0_pos)
        active_orders = self._active_orders(controller)

        # Initialize bot tracking
        for bot_id in bot_ids:
            if bot_id not in self.bot_orders:
                self.bot_orders[bot_id] = None
            if bot_id not in self.bot_plans:
                self.bot_plans[bot_id] = []
            if bot_id not in self.bot_plan_idx:
                self.bot_plan_idx[bot_id] = 0
            if bot_id not in self.bot_assembly_counters:
                self.bot_assembly_counters[bot_id] = None
            if bot_id not in self.bot_cooker_tiles:
                self.bot_cooker_tiles[bot_id] = None
            if bot_id not in self.bot_prep_counters:
                self.bot_prep_counters[bot_id] = None

        # Each bot works independently on their own recipe
        for bot_id in bot_ids:
            bot = controller.get_bot_state(bot_id)
            if not bot:
                continue

            # Check if bot's current order still exists
            current_order_id = self.bot_orders.get(bot_id)
            if current_order_id is not None:
                order = next((o for o in active_orders if o.get("order_id") == current_order_id), None)
                if order is None:
                    # Order expired or completed, reset this bot completely
                    self.bot_orders[bot_id] = None
                    self.bot_plans[bot_id] = []
                    self.bot_plan_idx[bot_id] = 0
                    self.bot_assembly_counters[bot_id] = None
                    self.bot_cooker_tiles[bot_id] = None
                    self.bot_prep_counters[bot_id] = None

            # Check if bot needs a new order
            current_plan = self.bot_plans.get(bot_id, [])
            current_idx = self.bot_plan_idx.get(bot_id, 0)

            if not current_plan or current_idx >= len(current_plan):
                # Clear hands before starting new order
                if self._clear_hands(controller, bot_id):
                    continue

                # Assign new order to this bot
                assigned_order_ids = set(self.bot_orders.values()) - {None}
                available_orders = [o for o in active_orders
                                   if o.get("order_id") not in assigned_order_ids
                                   and o.get("order_id") not in self.completed_orders]

                if not available_orders:
                    # No available orders, try to steal from completed or reuse
                    available_orders = [o for o in active_orders
                                       if o.get("order_id") not in assigned_order_ids]

                if available_orders:
                    # Choose best order for this bot
                    next_order = self._choose_order(controller, 1)
                    if next_order and next_order.get("order_id") not in assigned_order_ids:
                        self.bot_orders[bot_id] = next_order.get("order_id")
                        self.bot_plans[bot_id] = self._single_build_plan(next_order)
                        self.bot_plan_idx[bot_id] = 0
                    else:
                        # Just pick first available
                        self.bot_orders[bot_id] = available_orders[0].get("order_id")
                        self.bot_plans[bot_id] = self._single_build_plan(available_orders[0])
                        self.bot_plan_idx[bot_id] = 0
                else:
                    continue

            # Execute current step of this bot's plan
            current_plan = self.bot_plans[bot_id]
            current_idx = self.bot_plan_idx[bot_id]

            if current_idx < len(current_plan):
                task = current_plan[current_idx]
                if self._single_step(controller, bot_id, task):
                    self.bot_plan_idx[bot_id] += 1

                    # Check if bot just completed their order
                    if self.bot_plan_idx[bot_id] >= len(current_plan):
                        if self.bot_orders[bot_id] is not None:
                            self.completed_orders.add(self.bot_orders[bot_id])
                        self.bot_orders[bot_id] = None
                        self.bot_plans[bot_id] = []
                        self.bot_plan_idx[bot_id] = 0
