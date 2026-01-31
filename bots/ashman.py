from collections import deque
from typing import Dict, List, Optional, Tuple, Set

from game_constants import FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food


class BotPlayer:
	"""Instruction-driven bot that follows a scripted task list."""

	def __init__(self, map_copy):
		self.map = map_copy
		self.bot_id: Optional[int] = None
		self.turn = 0
		self.log_paths: Dict[int, str] = {}

		self.tile_positions: Dict[str, List[Tuple[int, int]]] = {}
		self._index_tiles()

		self.assembly_counter: Optional[Tuple[int, int]] = None
		self.cooker_loc: Optional[Tuple[int, int]] = None

		# Task list: each step is a simple instruction with a target and optional item.
		self.plan: List[Dict[str, object]] = [
			{"action": "ensure_pan"},
			{"action": "buy", "item": FoodType.MEAT, "target": "SHOP"},
			{"action": "place", "target": "COUNTER"},
			{"action": "chop", "target": "COUNTER"},
			{"action": "pickup", "target": "COUNTER"},
			{"action": "place", "target": "COOKER"},
			{"action": "buy", "item": ShopCosts.PLATE, "target": "SHOP"},
			{"action": "place", "target": "COUNTER"},
			{"action": "buy", "item": FoodType.NOODLES, "target": "SHOP"},
			{"action": "add_food_to_plate", "target": "COUNTER"},
			{"action": "wait_cooked", "target": "COOKER"},
			{"action": "add_food_to_plate", "target": "COUNTER"},
			{"action": "pickup", "target": "COUNTER"},
			{"action": "submit", "target": "SUBMIT"},
			
			{"action": "buy", "item": FoodType.MEAT, "target": "SHOP"},
			{"action": "place", "target": "COUNTER"},
			{"action": "chop", "target": "COUNTER"},
			{"action": "pickup", "target": "COUNTER"},
			{"action": "place", "target": "COOKER"},
			{"action": "buy", "item": ShopCosts.PLATE, "target": "SHOP"},
			{"action": "place", "target": "COUNTER"},
			{"action": "buy", "item": FoodType.NOODLES, "target": "SHOP"},
			{"action": "add_food_to_plate", "target": "COUNTER"},
			{"action": "wait_cooked", "target": "COOKER"},
			{"action": "add_food_to_plate", "target": "COUNTER"},
			{"action": "pickup", "target": "COUNTER"},
			{"action": "submit", "target": "SUBMIT"},
			
		]
		self.step_index = 0

	# ----------------------
	# Map helpers
	# ----------------------
	def _index_tiles(self) -> None:
		for x in range(self.map.width):
			for y in range(self.map.height):
				tile = self.map.tiles[x][y]
				self.tile_positions.setdefault(tile.tile_name, []).append((x, y))

	def _nearest_tile(self, tile_name: str, from_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
		positions = self.tile_positions.get(tile_name, [])
		if not positions:
			return None
		fx, fy = from_pos
		best = None
		best_dist = 10**9
		for x, y in positions:
			dist = max(abs(fx - x), abs(fy - y))
			if dist < best_dist:
				best_dist = dist
				best = (x, y)
		return best

	def _get_target_pos(self, tile_name: str, bot_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
		if tile_name == "COUNTER":
			if self.assembly_counter is None:
				self.assembly_counter = self._nearest_tile("COUNTER", bot_pos)
			return self.assembly_counter
		if tile_name == "COOKER":
			if self.cooker_loc is None:
				self.cooker_loc = self._nearest_tile("COOKER", bot_pos)
			return self.cooker_loc
		return self._nearest_tile(tile_name, bot_pos)

	# ----------------------
	# Movement helpers
	# ----------------------
	def _bfs_next_step(
		self,
		controller: RobotController,
		start: Tuple[int, int],
		target_predicate,
		blocked: Set[Tuple[int, int]],
	) -> Optional[Tuple[int, int]]:
		queue = deque([(start, [])])
		visited = {start}
		w, h = self.map.width, self.map.height

		while queue:
			(cx, cy), path = queue.popleft()
			tile = controller.get_tile(controller.get_team(), cx, cy)
			if target_predicate(cx, cy, tile):
				if not path:
					return (0, 0)
				return path[0]

			for dx in [-1, 0, 1]:
				for dy in [-1, 0, 1]:
					if dx == 0 and dy == 0:
						continue
					nx, ny = cx + dx, cy + dy
					if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
						if (nx, ny) in blocked:
							continue
						if controller.get_map().is_tile_walkable(nx, ny):
							visited.add((nx, ny))
							queue.append(((nx, ny), path + [(dx, dy)]))
		return None

	def _get_blocked_positions(self, controller: RobotController, bot_id: int) -> Set[Tuple[int, int]]:
		blocked: Set[Tuple[int, int]] = set()
		for other_id in controller.get_team_bot_ids():
			if other_id == bot_id:
				continue
			other_state = controller.get_bot_state(other_id)
			if not other_state:
				continue
			blocked.add((other_state["x"], other_state["y"]))
		return blocked

	def _move_adjacent(self, controller: RobotController, bot_id: int, tx: int, ty: int) -> bool:
		bot_state = controller.get_bot_state(bot_id)
		if not bot_state:
			return False
		bx, by = bot_state["x"], bot_state["y"]
		blocked = self._get_blocked_positions(controller, bot_id)

		if max(abs(bx - tx), abs(by - ty)) <= 1:
			return True

		def is_adjacent(x, y, _tile):
			return max(abs(x - tx), abs(y - ty)) <= 1

		step = self._bfs_next_step(controller, (bx, by), is_adjacent, blocked)
		if step and (step[0] != 0 or step[1] != 0):
			controller.move(bot_id, step[0], step[1])
		return False

	# ----------------------
	# Task execution
	# ----------------------
	def _reset_to_meat(self) -> None:
		self.step_index = 1

	def _run_task(self, controller: RobotController, bot_id: int) -> None:
		if self.step_index >= len(self.plan):
			self.step_index = 0

		task = self.plan[self.step_index]
		action = task["action"]
		self._log(controller, bot_id, f"step={self.step_index} action={action} task={task}")

		bot_state = controller.get_bot_state(bot_id)
		if not bot_state:
			return
		bx, by = bot_state["x"], bot_state["y"]

		target_name = task.get("target")
		target_pos = None
		if isinstance(target_name, str):
			target_pos = self._get_target_pos(target_name, (bx, by))
			if target_pos is None:
				return

		# Special composite task: ensure there is a pan on the cooker.
		if action == "ensure_pan":
			self._log(controller, bot_id, "ensure_pan: checking cooker pan")
			if self.cooker_loc is None:
				self.cooker_loc = self._nearest_tile("COOKER", (bx, by))
			if self.cooker_loc is None:
				return
			kx, ky = self.cooker_loc
			tile = controller.get_tile(controller.get_team(), kx, ky)
			if tile and isinstance(tile.item, Pan):
				self._log(controller, bot_id, "ensure_pan: pan present, advancing")
				self.step_index += 1
				return

			holding = bot_state.get("holding")
			if holding and holding.get("type") == "Pan":
				if self._move_adjacent(controller, bot_id, kx, ky):
					if controller.place(bot_id, kx, ky):
						self._log(controller, bot_id, f"place pan at ({kx},{ky})")
						self.step_index += 1
				return

			shop_pos = self._get_target_pos("SHOP", (bx, by))
			if shop_pos is None:
				return
			sx, sy = shop_pos
			if self._move_adjacent(controller, bot_id, sx, sy):
				if controller.get_team_money() >= ShopCosts.PAN.buy_cost:
					if controller.buy(bot_id, ShopCosts.PAN, sx, sy):
						self._log(controller, bot_id, f"buy PAN at ({sx},{sy})")
			return

		# Standard tasks
		if isinstance(target_pos, tuple):
			tx, ty = target_pos
			if not self._move_adjacent(controller, bot_id, tx, ty):
				return
		else:
			tx = ty = None

		holding = bot_state.get("holding")

		if action == "buy":
			item = task.get("item")
			if item and isinstance(target_pos, tuple):
				if holding is None:
					if controller.buy(bot_id, item, tx, ty):
						self._log(controller, bot_id, f"buy {item} at ({tx},{ty})")
						self.step_index += 1
			return

		if action == "place":
			if isinstance(target_pos, tuple):
				if holding is not None and controller.place(bot_id, tx, ty):
					self._log(controller, bot_id, f"place at ({tx},{ty})")
					self.step_index += 1
			return

		if action == "pickup":
			if isinstance(target_pos, tuple):
				if holding is None and controller.pickup(bot_id, tx, ty):
					self._log(controller, bot_id, f"pickup at ({tx},{ty})")
					self.step_index += 1
			return

		if action == "chop":
			if isinstance(target_pos, tuple):
				if holding is None and controller.chop(bot_id, tx, ty):
					self._log(controller, bot_id, f"chop at ({tx},{ty})")
					self.step_index += 1
			return

		if action == "add_food_to_plate":
			if isinstance(target_pos, tuple):
				if controller.add_food_to_plate(bot_id, tx, ty):
					self._log(controller, bot_id, f"add_food_to_plate at ({tx},{ty})")
					self.step_index += 1
			return

		if action == "wait_cooked":
			if not isinstance(target_pos, tuple):
				return
			tile = controller.get_tile(controller.get_team(), tx, ty)
			if tile and isinstance(tile.item, Pan) and tile.item.food:
				food = tile.item.food
				if food.cooked_stage == 1:
					if holding is None and controller.take_from_pan(bot_id, tx, ty):
						self._log(controller, bot_id, f"take_from_pan at ({tx},{ty})")
						self.step_index += 1
				elif food.cooked_stage >= 2:
					if holding is None and controller.take_from_pan(bot_id, tx, ty):
						self._log(controller, bot_id, f"take_from_pan (burnt) at ({tx},{ty})")
						trash_pos = self._get_target_pos("TRASH", (bx, by))
						if trash_pos:
							ttx, tty = trash_pos
							if self._move_adjacent(controller, bot_id, ttx, tty):
								if controller.trash(bot_id, ttx, tty):
									self._log(controller, bot_id, f"trash at ({ttx},{tty})")
						self._reset_to_meat()
				return

			# No food in pan -> restart meat flow
			self._reset_to_meat()
			return

		if action == "submit":
			if isinstance(target_pos, tuple):
				active_orders = [o for o in controller.get_orders() if o.get("is_active")]
				if not active_orders:
					self._log(controller, bot_id, "no active orders; skipping submit")
					return
				if controller.submit(bot_id, tx, ty):
					self._log(controller, bot_id, f"submit at ({tx},{ty})")
					self.step_index = 0
			return

	def _get_log_path(self, controller: RobotController, bot_id: int) -> str:
		if bot_id in self.log_paths:
			return self.log_paths[bot_id]
		team_name = controller.get_team().name.lower()
		path = f"bot_actions_{team_name}_{bot_id}.log"
		self.log_paths[bot_id] = path
		return path

	def _write_log_line(self, controller: RobotController, bot_id: int, line: str) -> None:
		path = self._get_log_path(controller, bot_id)
		with open(path, "a", encoding="utf-8") as f:
			f.write(line + "\n")

	def _log(self, controller: RobotController, bot_id: int, msg: str) -> None:
		bot_state = controller.get_bot_state(bot_id)
		if not bot_state:
			return
		bx, by = bot_state["x"], bot_state["y"]
		line = f"[BOT] turn={self.turn} bot={bot_id} pos=({bx},{by}) {msg}"
		self._write_log_line(controller, bot_id, line)

	def play_turn(self, controller: RobotController) -> None:
		self.turn = controller.get_turn()
		bots = controller.get_team_bot_ids()
		if not bots:
			return

		for b_id in bots:
			self._write_log_line(controller, b_id, f"[BOT] turn={self.turn} begin")

		# only the lowest bot id is allowed to act
		self.bot_id = min(bots)
		self._log(controller, self.bot_id, "selected as primary bot")
		self._run_task(controller, self.bot_id)

		for b_id in bots:
			if b_id != self.bot_id:
				self._write_log_line(controller, b_id, f"[BOT] turn={self.turn} bot={b_id} idle")
			self._write_log_line(controller, b_id, f"[BOT] turn={self.turn} end")
