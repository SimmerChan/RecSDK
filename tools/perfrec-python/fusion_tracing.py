import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from operator import attrgetter
from typing import Any, Dict, List

import toml


class MxRecEvent:
    def __init__(self, log_line: str, event_name: str, pipe_id: int):
        timestamp_s = get_timestamp(log_line)
        duration_ms = get_duration(log_line, event_name)
        process_id = get_process_id(log_line)
        self.timestamp_start_us = timestamp_s * 1e6 - float(duration_ms) * 1e3
        self.duration_us = float(duration_ms) * 1e3
        self.timestamp_end_us = timestamp_s * 1e6
        self.process_id = process_id
        self.name = event_name
        self.pipe_id = pipe_id


def extract_events(
    log_path: str, event_names: Dict[str, str]
) -> Dict[int, Dict[str, List[MxRecEvent]]]:
    events: Dict[int, Dict[str, List[MxRecEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pipe_names = get_pipes()
    pipe_ids = defaultdict(int)
    for i, pipe in enumerate(pipe_names):
        pipe_ids[pipe] = i
    with open(log_path) as log:
        for line in log:
            for name, pipe in event_names.items():
                if name in line:
                    event = MxRecEvent(line, name, pipe_ids[pipe])
                    events[event.process_id][pipe].append(event)
    return events


def merge_multithread_timeline(events: List[MxRecEvent]) -> List[MxRecEvent]:
    group_by_name = {
        name: sorted(list(group), key=attrgetter("timestamp_start_us"))
        for name, group in groupby(events, key=attrgetter("name"))
    }
    merged = list()
    for events in group_by_name.values():
        if len(events) > 1:
            tmp_merged = list()
            while events:
                event = events.pop(0)
                if tmp_merged:
                    last_event: MxRecEvent = tmp_merged.pop()
                    if event.timestamp_start_us <= last_event.timestamp_end_us:
                        event.timestamp_start_us = min(
                            event.timestamp_start_us, last_event.timestamp_start_us
                        )
                        event.timestamp_end_us = max(
                            event.timestamp_end_us, last_event.timestamp_end_us
                        )
                        event.duration_us = (
                            event.timestamp_end_us - event.timestamp_start_us
                        )
                        tmp_merged.append(event)
                    else:
                        tmp_merged.append(last_event)
                        tmp_merged.append(event)
                else:
                    tmp_merged.append(event)
            merged.extend(tmp_merged)
        else:
            merged.extend(events)
    return merged


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


def get_pipes() -> List[str]:
    config = toml.load("config.toml")
    pipes = list()
    for pipe in config["mxrec"].keys():
        pipes.append(pipe)
    return pipes


class TracingMetaData:
    def __init__(self, name: str, pid: int, tid: int, ph: str, args: Dict[str, Any]):
        self.name = name
        self.pid = pid
        self.tid = tid
        self.ph = ph
        self.args = args


class TracingEvent:
    def __init__(self, mxrec_event: MxRecEvent):
        self.name = mxrec_event.name
        self.pid = mxrec_event.process_id
        self.tid = get_fake_tid(self.pid, mxrec_event.pipe_id)
        self.ts = mxrec_event.timestamp_start_us
        self.dur = mxrec_event.duration_us
        self.ph = "X"
        self.args = {}


def get_metadata(processes: List[int]) -> List[TracingMetaData]:
    res = list()
    pipes = get_pipes()
    for i, pid in enumerate(processes):
        metadata1 = TracingMetaData(
            "process_name", pid, 0, "M", {"name": f"MxRec process {i}"}
        )
        metadata2 = TracingMetaData(
            "process_sort_index", pid, 0, "M", {"sort_index": i}
        )
        res.append(metadata1)
        res.append(metadata2)
        for pipe_i, pipe in enumerate(pipes):
            pipe_metadata1 = TracingMetaData(
                "thread_name",
                pid,
                get_fake_tid(pid, pipe_i),
                "M",
                {"name": f"{pipe} {pid}"},
            )
            pipe_metadata2 = TracingMetaData(
                "thread_sort_index",
                pid,
                get_fake_tid(pid, pipe_i),
                "M",
                {"sort_index": pipe_i},
            )
            res.append(pipe_metadata1)
            res.append(pipe_metadata2)
    return res


def get_fake_tid(pid: int, pipe_id: int) -> int:
    return pid * 10 + pipe_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPU/NPU fusion tracing json."
    )
    parser.add_argument("debug_log", help="MxRec DEBUG level log flie path.")
    args = parser.parse_args()

    log_path = args.debug_log
    config = read_config()

    mxrec_events = extract_events(log_path, config)
    tracing = list()
    tracing.extend(get_metadata(list(mxrec_events.keys())))
    for process in mxrec_events.values():
        for events in process.values():
            merged = merge_multithread_timeline(events)
            tracing.extend([TracingEvent(event) for event in merged])

    with open("mxrec_tracing.json", "w") as file:
        json.dump(tracing, file, indent=4, default=lambda obj: obj.__dict__)


if __name__ == "__main__":
    main()
