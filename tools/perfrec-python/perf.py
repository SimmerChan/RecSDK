import argparse
import os
import subprocess
from collections import defaultdict
from typing import List

from tabulate import tabulate
import toml


def generate_flamegraph(perf_data: str, output_svg: str, flamegraph_path: str) -> None:
    """
    Generate a flamegraph from perf data.

    Args:
        perf_data (str): Path to the perf.data file.
        output_svg (str): Path to the output SVG file.
        flamegraph_path (str): Path to the Flamegraph scripts directory.
    """
    # Ensure perf script is available
    try:
        subprocess.run(["perf", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("Error: perf is not installed or not in PATH.")
        return

    # Ensure Flamegraph scripts are available
    stackcollapse_path = os.path.join(flamegraph_path, "stackcollapse-perf.pl")
    flamegraph_script_path = os.path.join(flamegraph_path, "flamegraph.pl")

    if not os.path.isfile(stackcollapse_path) or not os.path.isfile(
        flamegraph_script_path
    ):
        print(
            f"Error: Flamegraph scripts not found in the provided directory {flamegraph_path}."
        )
        return

    # Generate the folded stack output
    folded_output = perf_data + ".folded"
    with open(folded_output, "w") as f:
        script_output = subprocess.run(
            ["perf", "script", "-i", perf_data], check=True, stdout=subprocess.PIPE
        )
        subprocess.run(
            [stackcollapse_path], check=True, input=script_output.stdout, stdout=f
        )

    # Generate the flamegraph
    with open(output_svg, "w") as f:
        subprocess.run([flamegraph_script_path, folded_output], check=True, stdout=f)

    print(f"Flamegraph generated at {output_svg}")

    # Analyze the folded stack output
    analyze_folded_stack(folded_output)


class CallStack:
    def __init__(self):
        self.count = 0
        self.call_stacks = []

    def add_call_stacks(self, count: int, call_stack: str):
        self.count += count
        self.call_stacks.append(call_stack)


def analyze_folded_stack(folded_output: str) -> None:
    """
    Analyzes the folded stack output to find functions with significant sample counts.

    Args:
        folded_output (str): Path to the folded stack output file.
    """

    function_counts = defaultdict(CallStack)
    total_count = 0

    # Read the folded stack output
    with open(folded_output, "r") as f:
        for line in f:
            parts = line.strip().rsplit(
                " ", 1
            )  # Use rsplit to handle function names with spaces
            count = int(parts[-1])
            call_stack_str = parts[0]
            stack = parts[0].split(";")
            function_counts[stack[-1]].add_call_stacks(count, call_stack_str)
            total_count += count

    config = read_config()

    # Filter and display functions with more than 5% total count
    threshold = total_count * config.threshold
    results = [
        (func, call_stack)
        for func, call_stack in function_counts.items()
        if call_stack.count >= threshold and func not in config.ignores
    ]

    # Sort results by count in descending order
    results.sort(key=lambda x: x[1].count, reverse=True)

    # Prepare data for tabulate
    # Write call stacks to file
    table_data = []
    with open("call_stacks.txt", "w") as f:
        for func, call_stack in results:
            percentage = (call_stack.count / total_count) * 100
            table_data.append(
                [limit_line(func, 50), call_stack.count, f"{percentage:.2f}%"]
            )
            stacks = [stk + "\n" for stk in call_stack.call_stacks]
            f.writelines(
                [
                    f"func_name: {func}\n",
                    f"percetage: {percentage:.2f}%\n",
                    "call_stacks:\n",
                ]
                + stacks
                + ["\n\n"]
            )

    # Print the results using tabulate
    print("\nFunctions with more than 5% of total samples:\n")
    headers = ["Function", "Count", "Percentage"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def limit_line(input: str, line_length: int) -> str:
    """
    Limits the length of a line to a specified number of characters, adding line breaks if necessary.

    Args:
        input (str): The input string.
        line_length (int): The maximum line length.

    Returns:
        str: The formatted string with line breaks.
    """
    if line_length >= len(input):
        return input
    limited_str = ""
    if line_length > 0:
        count = 0
        for c in input:
            if count >= line_length:
                limited_str += "\n"
                count = 0
            limited_str += c
            count += 1
    return limited_str


class PerfConfig:
    """
    Configuration from `config.toml`.
    """

    def __init__(self, threshold: int, ignores: List[str]):
        self.threshold = threshold
        self.ignores = set(ignores)


def read_config() -> PerfConfig:
    """
    Reads configs related to `perf` from the configuration file.

    Returns:
        PerfConfig: Configuration class.
    """
    config = toml.load("config.toml")
    perf_config = config["perf"]
    return PerfConfig(perf_config["threshold"], perf_config["ignores"])


def main():
    """
    Main function to parse arguments and generate a flamegraph.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Flamegraph from perf.data."
    )
    parser.add_argument(
        "--perf_data", help="Path to the perf.data file.", required=True
    )
    parser.add_argument(
        "--output_svg",
        help="Path to the output SVG file. (default: flamegraph.svg)",
        required=False,
        default="flamegraph.svg",
    )
    parser.add_argument(
        "--flamegraph_path",
        help="Path to the Flamegraph Perl scripts directory.",
        required=True,
    )
    args = parser.parse_args()

    generate_flamegraph(args.perf_data, args.output_svg, args.flamegraph_path)


if __name__ == "__main__":
    main()
