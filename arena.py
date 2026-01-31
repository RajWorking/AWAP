import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from game import Game  # noqa: E402
from game_constants import Team, GameConstants  # noqa: E402


@dataclass
class MatchResult:
    map_name: str
    red_bot: str
    blue_bot: str
    red_money: int
    blue_money: int
    winner: Optional[str]
    reason: str


def list_files(folder: str, suffix: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    items = []
    for name in os.listdir(folder):
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.endswith(suffix):
            items.append(path)
    return sorted(items)


def bot_label(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


def map_label(path: str) -> str:
    return os.path.basename(path)


def resolve_bot_arg(bot_arg: str, bots: List[str]) -> Optional[str]:
    if not bot_arg:
        return None
    for path in bots:
        if bot_label(path) == bot_arg:
            return path
    if bot_arg.endswith(".py"):
        for path in bots:
            if os.path.normpath(path) == os.path.normpath(bot_arg):
                return path
    return None


def resolve_map_arg(map_arg: str, maps: List[str]) -> Optional[str]:
    if not map_arg:
        return None
    for path in maps:
        if map_label(path) == map_arg:
            return path
    if map_arg.endswith(".txt"):
        for path in maps:
            if os.path.normpath(path) == os.path.normpath(map_arg):
                return path
    else:
        candidate = f"{map_arg}.txt"
        for path in maps:
            if map_label(path) == candidate:
                return path
    return None


def run_match(
    red_bot_path: str,
    blue_bot_path: str,
    map_path: str,
    *,
    turn_limit: int,
    per_turn_timeout_s: float,
) -> MatchResult:
    game = Game(
        red_bot_path=red_bot_path,
        blue_bot_path=blue_bot_path,
        map_path=map_path,
        replay_path=None,
        render=False,
        turn_limit=turn_limit,
        per_turn_timeout_s=per_turn_timeout_s,
    )
    try:
        winner_by_crash = game.run_game()
        red_money = game.game_state.get_team_money(Team.RED)
        blue_money = game.game_state.get_team_money(Team.BLUE)
    finally:
        game.close()

    winner: Optional[str] = None
    reason = "score"
    if winner_by_crash == Team.RED:
        winner = bot_label(red_bot_path)
        reason = "blue_crash"
    elif winner_by_crash == Team.BLUE:
        winner = bot_label(blue_bot_path)
        reason = "red_crash"
    else:
        if red_money > blue_money:
            winner = bot_label(red_bot_path)
        elif blue_money > red_money:
            winner = bot_label(blue_bot_path)
        else:
            reason = "draw"

    return MatchResult(
        map_name=map_label(map_path),
        red_bot=bot_label(red_bot_path),
        blue_bot=bot_label(blue_bot_path),
        red_money=red_money,
        blue_money=blue_money,
        winner=winner,
        reason=reason,
    )


def update_leaderboard(
    leaderboard: Dict[str, Dict[str, int]],
    result: MatchResult,
) -> None:
    for bot in (result.red_bot, result.blue_bot):
        leaderboard.setdefault(bot, {"wins": 0, "losses": 0, "draws": 0, "games": 0})

    leaderboard[result.red_bot]["games"] += 1
    leaderboard[result.blue_bot]["games"] += 1

    if result.winner is None:
        leaderboard[result.red_bot]["draws"] += 1
        leaderboard[result.blue_bot]["draws"] += 1
        return

    if result.winner == result.red_bot:
        leaderboard[result.red_bot]["wins"] += 1
        leaderboard[result.blue_bot]["losses"] += 1
    else:
        leaderboard[result.blue_bot]["wins"] += 1
        leaderboard[result.red_bot]["losses"] += 1


def format_match_log(result: MatchResult) -> str:
    if result.winner is None:
        outcome = "DRAW"
        winner = "NONE"
        loser = "NONE"
    else:
        outcome = "WIN"
        winner = result.winner
        loser = result.blue_bot if result.winner == result.red_bot else result.red_bot
    return (
        f"[MATCH] map={result.map_name} winner={winner} loser={loser} "
        f"red={result.red_bot} blue={result.blue_bot} outcome={outcome} "
        f"red_money={result.red_money} blue_money={result.blue_money} "
        f"reason={result.reason}"
    )


def format_leaderboard(leaderboard: Dict[str, Dict[str, int]]) -> List[str]:
    rows: List[Tuple[str, int, int, int, int, float]] = []
    for bot, stats in leaderboard.items():
        games = stats["games"]
        wins = stats["wins"]
        losses = stats["losses"]
        draws = stats["draws"]
        win_rate = wins / games if games else 0.0
        rows.append((bot, wins, losses, draws, games, win_rate))

    rows.sort(key=lambda r: (-r[1], -r[3], r[2], r[0]))

    lines = ["[LEADERBOARD]"]
    for idx, (bot, wins, losses, draws, games, win_rate) in enumerate(rows, start=1):
        lines.append(
            f"{idx}. {bot} wins={wins} losses={losses} draws={draws} games={games} win_rate={win_rate:.3f}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots-dir", default="bots", help="directory containing bot .py files")
    ap.add_argument("--maps-dir", default="good_maps", help="directory containing map .txt files")
    ap.add_argument("--turns", type=int, default=GameConstants.TOTAL_TURNS, help="turn limit per game")
    ap.add_argument("--timeout", type=float, default=0.5, help="per-turn timeout seconds per bot")
    ap.add_argument("--log", default="arena.log", help="path to output log file")
    ap.add_argument(
        "--bot",
        default=None,
        help="optional bot name (without .py) or path; if set, run only this bot against all others",
    )
    ap.add_argument(
        "--map",
        default=None,
        help="optional map name (with .txt) or path; if set, run only this map",
    )
    args = ap.parse_args()

    bots = list_files(args.bots_dir, ".py")
    maps = list_files(args.maps_dir, ".txt")

    if not bots:
        print(f"[ARENA] No bots found in {args.bots_dir}")
        return 1
    if not maps:
        print(f"[ARENA] No maps found in {args.maps_dir}")
        return 1

    focus_bot_path = None
    if args.bot:
        focus_bot_path = resolve_bot_arg(args.bot, bots)
        if focus_bot_path is None:
            print(f"[ARENA] Bot not found: {args.bot}")
            return 1
        if len(bots) < 2:
            print("[ARENA] Need at least two bots to run matches")
            return 1

    focus_map_path = None
    if args.map:
        focus_map_path = resolve_map_arg(args.map, maps)
        if focus_map_path is None:
            print(f"[ARENA] Map not found: {args.map}")
            return 1

    leaderboard: Dict[str, Dict[str, int]] = {}
    match_logs: List[str] = []

    if focus_bot_path:
        opponents = [b for b in bots if b != focus_bot_path]
        if not opponents:
            print("[ARENA] No opponents found for selected bot")
            return 1
        pairings = [(focus_bot_path, other) for other in opponents]
    else:
        pairings = []
        for i, red_bot in enumerate(bots):
            for blue_bot in bots[i + 1 :]:
                pairings.append((red_bot, blue_bot))

    map_list = [focus_map_path] if focus_map_path else maps
    for map_path in map_list:
        for red_bot, blue_bot in pairings:
            result = run_match(
                red_bot,
                blue_bot,
                map_path,
                turn_limit=args.turns,
                per_turn_timeout_s=args.timeout,
            )
            update_leaderboard(leaderboard, result)
            match_logs.append(format_match_log(result))

    leaderboard_lines = format_leaderboard(leaderboard)

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("[ARENA] Bots: " + ", ".join(bot_label(b) for b in bots) + "\n")
        if focus_bot_path:
            f.write("[ARENA] Focus bot: " + bot_label(focus_bot_path) + "\n")
        if focus_map_path:
            f.write("[ARENA] Focus map: " + map_label(focus_map_path) + "\n")
        f.write("[ARENA] Maps: " + ", ".join(map_label(m) for m in maps) + "\n")
        for line in match_logs:
            f.write(line + "\n")
        for line in leaderboard_lines:
            f.write(line + "\n")

    print(f"[ARENA] wrote log to {args.log}")
    for line in leaderboard_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
