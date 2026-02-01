#!/usr/bin/env python3
"""
Convert a JSON map (tiles + bots + orders) into the .txt layout format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


CHAR_BY_TILE_NAME: Dict[str, str] = {
    "FLOOR": ".",
    "WALL": "#",
    "COUNTER": "C",
    "COOKER": "K",
    "SINK": "S",
    "SINKTABLE": "T",
    "TRASH": "R",
    "SUBMIT": "U",
    "SHOP": "$",
    "BOX": "B",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_layout(tiles: List[List[dict]]) -> List[List[str]]:
    if not tiles:
        raise ValueError("tiles is empty")
    width = len(tiles[0])
    if any(len(r) != width for r in tiles):
        raise ValueError("tiles has inconsistent row widths")

    layout: List[List[str]] = []
    for row in tiles:
        out_row: List[str] = []
        for cell in row:
            name = cell.get("tile_name")
            if name not in CHAR_BY_TILE_NAME:
                raise ValueError(f'Unknown tile_name "{name}"')
            out_row.append(CHAR_BY_TILE_NAME[name])
        layout.append(out_row)
    return layout


def _place_bots(layout: List[List[str]], bots: List[dict]) -> None:
    height = len(layout)
    width = len(layout[0]) if height else 0
    for bot in bots:
        x = bot.get("x")
        y = bot.get("y")
        if x is None or y is None:
            raise ValueError(f"Bot missing x/y: {bot}")
        file_row = height - 1 - y
        if not (0 <= x < width and 0 <= file_row < height):
            raise ValueError(f"Bot out of bounds: {bot}")
        layout[file_row][x] = "b"


def _format_orders(orders: List[dict]) -> List[str]:
    if not orders:
        return []

    lines = ["", "ORDERS:"]
    for o in orders:
        start = o.get("created_turn")
        end = o.get("expires_turn")
        required = o.get("required", [])
        if start is None or end is None:
            raise ValueError(f"Order missing created_turn/expires_turn: {o}")
        duration = end - start
        if duration < 0:
            raise ValueError(f"Order has negative duration: {o}")
        required_csv = ",".join(required)
        reward = o.get("reward", 5)
        penalty = o.get("penalty", 2)
        lines.append(
            f"start={start}  duration={duration}  required={required_csv}  reward={reward} penalty={penalty}"
        )
    return lines


def json_to_txt(data: dict) -> str:
    tiles = data.get("tiles")
    if tiles is None:
        raise ValueError('JSON missing "tiles"')

    layout = _render_layout(tiles)

    bots = data.get("bots", [])
    if bots:
        _place_bots(layout, bots)

    lines = ["".join(r) for r in layout]
    lines.extend(_format_orders(data.get("orders", [])))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSON map to .txt format.")
    parser.add_argument("--input", type=Path, help="Path to JSON map file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .txt path (defaults to input with .txt extension)",
    )
    args = parser.parse_args()

    data = _load_json(args.input)
    out_text = json_to_txt(data)

    out_path = args.output or args.input.with_suffix(".txt")
    out_path.write_text(out_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
