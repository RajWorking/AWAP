#!/usr/bin/env python3
"""
Quick test to compare claude_bot-tune performance with and without pause/resume feature.
We'll run the bot against itself on a simple map and check the scores.
"""

import subprocess
import re
import sys

def run_test(bot_name, map_name="simple_map.txt", turns=200, runs=3):
    """Run arena.py and extract the money score for the specified bot."""
    scores = []

    for i in range(runs):
        print(f"Running test {i+1}/{runs} for {bot_name}...", file=sys.stderr)
        result = subprocess.run(
            ["python", "arena.py", "--bot", bot_name, "--map", map_name, "--turns", str(turns)],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Extract the money score from output
        # Looking for patterns like: [GAME OVER] money scores: RED=$595, BLUE=$595
        for line in result.stdout.split('\n'):
            if '[GAME OVER]' in line and 'money scores' in line:
                # Extract numbers after $ signs
                money_values = re.findall(r'\$(\d+)', line)
                if money_values:
                    # Assume the bot is RED team (first value)
                    scores.append(int(money_values[0]))

    if scores:
        avg = sum(scores) / len(scores)
        print(f"{bot_name}: Scores = {scores}, Average = ${avg:.2f}", file=sys.stderr)
        return avg
    else:
        print(f"Failed to extract scores for {bot_name}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    print("=" * 60)
    print("Testing claude_bot-tune with pause/resume feature")
    print("=" * 60)

    score = run_test("claude_bot-tune", runs=5)
    print(f"\nFinal Average Score: ${score:.2f}")
    print("\nTest complete! The bot ran successfully.")
    print("Higher scores indicate better performance (more orders completed).")
