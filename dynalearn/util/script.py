import os
import time as tt

from itertools import product

PRECISION = 4


def launch(
    name,
    path,
    path_to_script,
    config={},
    command="sbatch",
    time="2:00:00",
    memory="24G",
    account="def-aallard",
    modules_to_load=[],
    source_path=None,
    devices="cpu",
    verbose=0,
):
    path_to_data = os.path.join(path, "full_data")
    path_to_best = os.path.join(path, "best")
    path_to_summaries = os.path.join(path, "summaries")
    path_to_outputs = os.path.join(path, "outputs")

    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    if not os.path.exists(path_to_best):
        os.makedirs(path_to_best)
    if not os.path.exists(path_to_summaries):
        os.makedirs(path_to_summaries)
    if not os.path.exists(path_to_outputs):
        os.makedirs(path_to_outputs)

    s = devices.split(" ")
    if s[0].isdigit():
        num_devices = int(s[0])
        device = s[1]
    else:
        num_devices = 1
        device = s[0]
    script = "#!/bin/bash\n"
    if command == "sbatch":
        script += f"#SBATCH --account={account}\n"
        script += f"#SBATCH --time={time}\n"
        script += f"#SBATCH --job-name={name}\n"
        script += f"#SBATCH --output={os.path.join(path_to_outputs, name)}.out\n"
        script += f"#SBATCH --mem={memory}\n"
        if device == "cpu":
            script += f"#SBATCH --cpus-per-task={num_devices}\n"
        elif device == "gpu":
            script += f"#SBATCH --gres=gpu:{num_devices}\n"
        script += "\n"
        script += f"module {' '.join(modules_to_load)}\n"
        script += f"source {source_path}\n"
    script += f"python {path_to_script}"

    for k, v in config.items():
        if isinstance(v, tuple):
            script += f" --{k} {' '.join(v)}"
        else:
            script += f" --{k} {v}"
    script += f" --name {name}"
    script += f" --path_to_data {path_to_data}"
    script += f" --path_to_best {path_to_best}"
    script += f" --path_to_summary {path_to_summaries}"
    script += f" --verbose {verbose}"

    if command is "sbatch":
        script += "deactivate\n"

    path_to_script = f"{int(tt.time())}.sh"
    with open(path_to_script, "w") as f:
        f.write(script)

    os.system(f"{command} {path_to_script}")
    os.remove(path_to_script)


def launch_scan(
    name,
    path_to_script,
    path_to_data,
    config={},
    command="sbatch",
    time="2:00:00",
    memory="24G",
    account="def-aallard",
    modules_to_load=[],
    source_path=None,
    verbose=0,
    devices="cpu",
):
    local_config = {}
    labels_to_scan = []
    values_to_scan = []
    suffix = ""
    for k, v in config.items():
        if isinstance(v, list):
            labels_to_scan.append(k)
            values_to_scan.append(v)

        else:
            local_config[k] = v

    if local_config == config:
        launch(
            name,
            path_to_script,
            path_to_data,
            config=local_config,
            command=command,
            time=time,
            memory=memory,
            account=account,
            modules_to_load=modules_to_load,
            source_path=source_path,
            verbose=verbose,
            devices=devices,
        )
    else:
        for vals in product(*values_to_scan):
            suffix = ""
            for k, v in zip(labels_to_scan, vals):
                local_config[k] = v
                n = k.split("_")
                prefix = "".join([nn[0] for nn in n])
                if isinstance(v, str):
                    suffix += f"{v}-"
                elif isinstance(v, int):
                    suffix += f"{prefix}{v}-"
                elif isinstance(v, float):
                    suffix += f"{prefix}{round(v, PRECISION)}-"
            suffix = suffix[:-1]
            launch(
                name + "-" + suffix,
                path_to_script,
                path_to_data,
                config=local_config,
                command=command,
                time=time,
                memory=memory,
                account=account,
                modules_to_load=modules_to_load,
                source_path=source_path,
                verbose=verbose,
                devices=devices,
            )
