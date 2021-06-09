

def read_config(cfg_path="config.csv"):
    cfg = dict()
    with open(cfg_path, "r") as file:
        for line in file.readlines():
            line = line.strip()
            line = line.split(",")
            key = line[0]
            if key.startswith("var_"):
                data = float(line[1])
            else:
                data = line[1]
            cfg[key] = data
    return cfg


def save_config(cfg: dict):
    cfg_path = cfg["config_file_path"]
    with open(cfg_path, "w") as file:
        for key in cfg.keys():
            line = str(key) + "," + str(cfg[key]) + "\n"
            file.write(line)
    return True
