import argparse
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import toml


class MxRecEvent:
    def __init__(self, log_line: str, event_name: str):
        timestamp_s = get_timestamp(log_line)
        duration_ms = get_duration(log_line, event_name)
        process_id = get_process_id(log_line)
        self.timestamp_start_us = timestamp_s * 1e6 - float(duration_ms) * 1e3
        self.duration_us = float(duration_ms) * 1e3
        self.process_id = process_id
        self.name = event_name


def extract_events(
    log_path: str, event_names: Dict[str, str]
) -> Dict[str, List[MxRecEvent]]:
    events = defaultdict(list)
    with open(log_path) as log:
        for line in log:
            for name in event_names.keys():
                if name in line:
                    events[event_names[name]].append(MxRecEvent(line, name))
    return events


def get_timestamp(log_line: str) -> float:
    pattern = r"\[(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}\.\d+)\]"
    match = re.search(pattern, log_line)
    if match:
        date_time_str = match.group(1)
        date_time_format = "%Y/%m/%d %H:%M:%S.%f"
        # Parse the date-time string into a datetime object
        date_time_obj = datetime.strptime(date_time_str, date_time_format)
        # Convert the datetime object to a timestamp
        return date_time_obj.timestamp()
    else:
        raise RuntimeError(f"There is no time in log: {log_line}")


def get_duration(log_line: str, event_name: str) -> int:
    pattern = event_name + r".*:\s*(\d+)"
    match = re.search(pattern, log_line)
    if match:
        duration_ms = match.group(1)
        return int(duration_ms)
    else:
        raise RuntimeError(f"There is no event: {event_name}, log: {log_line}")


def get_process_id(log_line: str) -> int:
    pattern = r"process_id:\s*(\d+)"
    match = re.search(pattern, log_line)
    if match:
        process_id = match.group(1)
        return int(process_id)
    else:
        raise RuntimeError(f"There is no process_id in log: {log_line}")


def read_config() -> Dict[str, str]:
    config = toml.load("config.toml")
    mxrec_config = defaultdict(str)
    for pipe, event_list in config["mxrec"].items():
        for event in event_list:
            mxrec_config[event] = pipe
    return mxrec_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPU/NPU fusion tracing json."
    )
    parser.add_argument("debug_log", help="MxRec DEBUG level log.")
    args = parser.parse_args()

    log_path = args.debug_log
    config = read_config()
    print(extract_events(log_path, config))


if __name__ == "__main__":
    main()
