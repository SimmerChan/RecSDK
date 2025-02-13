from absl import app, flags
import multiprocessing
import os
import re
import subprocess


FLAGS = flags.FLAGS

flags.DEFINE_string("args", None, "Arguments to pass to the script")


def run_script(base_device, total_devices, base_args):
    script_directory = os.path.join(os.path.dirname(__file__))

    main_script_path = os.path.join(script_directory, "main.py")

    cmd = ["python3", main_script_path, f"--device={base_device}", f"--total_devices={total_devices}"]
    cmd.extend(base_args)
    subprocess.run(cmd)


def parse_args(args_string):
    base_args = []
    total_devices = 1
    base_device = "npu"
    for arg_v in re.split(r"\s+--", args_string.strip()):
        arg_v = arg_v.strip("--")
        arg, v = re.split(r"\s+|=", arg_v)
        if arg == "base_device":
            base_device = v
        elif arg == "total_devices":
            total_devices = int(v)
        base_args.append(f"--{arg}={v}")
    return base_device, total_devices, base_args


def start_processes(base_device, total_devices, base_args):

    base_args, total_devices, base_args = parse_args(FLAGS.args)

    processes = []

    for i in range(total_devices):
        sub_base_device = f"{base_device}:{i}"
        p = multiprocessing.Process(target=run_script, args=(sub_base_device, total_devices, base_args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(start_processes)
