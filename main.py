import os
import yaml
import inspect
import importlib
from types import SimpleNamespace
import argparse
import numpy as np

from eval import eval



def save_config(args, original_config_path):
    if original_config_path.endswith((".yaml", ".yml")):
        with open(original_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        save_path = os.path.join(args.exp_name, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
    elif original_config_path.endswith(".json"):
        import json
        with open(original_config_path, "r") as f:
            config_dict = json.load(f)
        save_path = os.path.join(args.exp_name, "config.json")
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    else:
        raise ValueError("Unsupported config file format for saving.")

    print(f"Config file saved to: {save_path}")


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    else:
        return obj


def load_config(config_path):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must end with .yaml, .yml, or .json")
    return config


def instantiate_model_from_config(args, cfg):
    model_module = importlib.import_module(cfg.model.module)
    target_name = cfg.model.name

    if hasattr(model_module, target_name):
        model_class = getattr(model_module, target_name)
    else:
        model_class = None
        for _, obj in inspect.getmembers(model_module, inspect.isclass):
            if obj.__name__ == target_name:
                model_class = obj
                break

        if model_class is None:
            raise AttributeError()

    model = model_class(args, cfg)
    return model


def main():
    parser = argparse.ArgumentParser(description="Argument parser for specifying dataset, model, and training configurations")

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--split_interval", type=int)

    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
        print(f"Folder '{args.exp_name}' created.")
    else:
        print(f"Folder '{args.exp_name}' already exists.")

    config = load_config(args.config)
    cfg = dict_to_namespace(config)
    save_config(args, args.config)

    model = instantiate_model_from_config(args, cfg)

    eval(args, model)


if __name__ == "__main__":
    main()

