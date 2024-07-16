import argparse
import re
from datetime import datetime


class MxRecEvent:
    def __init__(self, log_line: str, event_name: str):
        timestamp_s = get_timestamp(log_line)
        duration_ms = get_duration(log_line, event_name)
        self.timestamp_start_us = timestamp_s * 1e6 - float(duration_ms) * 1e3
        self.duration_us = float(duration_ms) * 1e3


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


def get_thread_id(log_line: str) -> int:
    pattern = r"thread_id:\s*(\d+)"
    match = re.search(pattern, log_line)
    if match:
        thread_id = match.group(1)
        return int(thread_id)
    else:
        raise RuntimeError(f"There is no thread_id in log: {log_line}")


def main():
    log = "[1,0]<stdout>:[MxRec][2024/7/15 16:28:5.148569] some other contents getAndSendTensorsTC(ms): 166."
    print(get_timestamp(log))
    print(get_duration(log, "getAndSendTensor"))
    event = MxRecEvent(log, "getAndSendTensors")
    print(event.timestamp_start_us, event.duration_us)

    parser = argparse.ArgumentParser(description="Generate CPU/NPU fusion tracing json.")
    parser.add_argument("debug_log", help="MxRec DEBUG level log.")
    args = parser.parse_args()

    thread_ids = set()
    with open(args.debug_log) as log:
        for line in log:
            if "getAndSendTensorsTC" in line:
                thread_ids.add(get_thread_id(line))
    print(thread_ids)

if __name__ == "__main__":
    main()
