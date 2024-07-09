import argparse
import os
import subprocess
from collections import defaultdict


def generate_flamegraph(perf_data, output_svg, flamegraph_path):
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


def analyze_folded_stack(folded_output):
    function_counts = defaultdict(int)
    total_count = 0

    # Read the folded stack output
    with open(folded_output, "r") as f:
        for line in f:
            parts = line.strip().rsplit(
                " ", 1
            )  # Use rsplit to handle function names with spaces
            count = int(parts[-1])
            stack = parts[0].split(";")
            for function in stack:
                function_counts[function] += count
            total_count += count

    # Filter and display functions with more than 5% total count
    threshold = total_count * 0.05
    results = [
        (func, count) for func, count in function_counts.items() if count >= threshold
    ]

    # Sort results by count in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Print the results in an ASCII table format
    print("\nFunctions with more than 5% of total samples:\n")
    print(f"{'Function':<40} {'Count':<10} {'Percentage':<10}")
    print("=" * 60)
    for func, count in results:
        percentage = (count / total_count) * 100
        print(f"{func:<40} {count:<10} {percentage:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Flamegraph from perf.data")
    parser.add_argument("perf_data", help="Path to the perf.data file")
    parser.add_argument("output_svg", help="Path to the output SVG file")
    parser.add_argument(
        "flamegraph_path", help="Path to the Flamegraph scripts directory"
    )
    args = parser.parse_args()

    generate_flamegraph(args.perf_data, args.output_svg, args.flamegraph_path)


if __name__ == "__main__":
    main()
