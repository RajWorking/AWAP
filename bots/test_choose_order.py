# test_choose_order.py
from enum import Enum, auto

# --- minimal FoodType mock ---
class FoodType(Enum):
    NOODLES = auto()
    MEAT = auto()
    VEGGIE = auto()

    @property
    def buy_cost(self):
        return {
            FoodType.NOODLES: 5,
            FoodType.MEAT: 12,
            FoodType.VEGGIE: 3,
        }[self]

    @property
    def can_cook(self):
        return self in {FoodType.MEAT}

    @property
    def can_chop(self):
        return self in {FoodType.VEGGIE}


# --- minimal controller mock ---
class MockController:
    def __init__(self, orders, turn=10, money=100):
        self._orders = orders
        self._turn = turn
        self._money = money

    def get_orders(self, team):
        return self._orders

    def get_turn(self):
        return self._turn

    def get_team_money(self, team):
        return self._money

    def get_team(self):
        return 0


# --- import your BotPlayer here ---
from bot_player import BotPlayer   # adjust filename if needed


# --- fake map (BotPlayer __init__ needs it) ---
class DummyMap:
    width = 1
    height = 1
    tiles = [[None]]


# --- test orders ---
orders = [
    {
        "order_id": 1,
        "is_active": True,
        "required": ["NOODLES", "MEAT"],
        "reward": 120,
        "penalty": 30,
        "expires_turn": 200,
    },
    {
        "order_id": 2,
        "is_active": True,
        "required": ["VEGGIE"],
        "reward": 20,
        "penalty": 5,
        "expires_turn": 200,
    },
    {
        "order_id": 3,
        "is_active": True,
        "required": ["NOODLES"],
        "reward": 5,      # intentionally bad
        "penalty": 0,
        "expires_turn": 200,
    },
]

controller = MockController(orders)
bot = BotPlayer(DummyMap())

chosen = bot._choose_order(controller)

print("Chosen order:")
print(chosen)
