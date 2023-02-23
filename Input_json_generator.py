import os
import yaml
import json


def make_input_json(application_dir):
    filename = "SPACE4AI-D.yaml"
    filepath = os.path.join(application_dir, "space4ai-d", filename)

    with open(filepath) as file:
        content = yaml.load(file, Loader=yaml.Loader)
        methods = content["Methods"]
        seed = content["Seed"]
        verbose_level = content["VerboseLevel"]

    filename = "Input.json"
    filepath = os.path.join(application_dir, "space4ai-d", filename)

    with open(filepath, "w") as file:
        json.dump({"Methods": methods, "Seed": seed, "VerboseLevel": verbose_level}, file, indent=4)
    return filepath

if __name__ == '__main__':
    make_input_json("Demo_project")
