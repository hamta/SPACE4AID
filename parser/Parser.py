import os
import yaml
import sys
from classes.Logger import Logger
import shutil

class Parser:

    def __init__(self, application_dir, who, directory="", log=Logger()):
        self.logger = log
        self.application_dir = application_dir
        self.error = Logger(stream=sys.stderr, verbose=1, error=True)
        folder_path = "common_config"
        self.common_config_path = os.path.join(application_dir, folder_path)
        self.is_degraded()
        if who.lower() in ["s4ai-d", "space4ai-d", "s4aid", "space4aid"]:
            self.who = "SPACE4AI-D"
            folder_path = "aisprint/designs"
            self.component_partitions_path = os.path.join(application_dir, folder_path)
            folder_path = "space4ai-d"
            self.space4aid_path = os.path.join(application_dir, folder_path)
            folder_path = "oscarp"
            self.oscarp_path = os.path.join(application_dir, folder_path)
            folder_path = "space4ai-r"
            self.space4air_path = os.path.join(application_dir, folder_path)
            folder_path = "aisprint/deployments/optimal_deployment"
            self.optimal_deployment_path = os.path.join(application_dir, folder_path)

        elif who.lower() in ["s4ai-r", "space4ai-r", "s4air", "space4air"]:
            self.who = "SPACE4AI-R"
            if self.degraded:
                if directory != "":
                    Dir = os.path.join(application_dir, directory)
                    if os.path.exists(Dir):
                        self.component_partitions_path = Dir
                        self.common_config_path = Dir
                        self.space4aid_path = Dir
                        self.oscarp_path = Dir
                        self.space4air_path = Dir
                        self.optimal_deployment_path = Dir

                    else:
                        self.error.log("The directory {} specified by SPACE4AI-R user is not exist.".format(Dir))
                        sys.exit(1)
                else:
                    self.error.log("A directory must be specified by SPACE4AI-R user.")
                    sys.exit(1)
            else:
                folder_path = "aisprint/designs"
                self.component_partitions_path = os.path.join(application_dir, folder_path)
                folder_path = "space4ai-d"
                self.space4aid_path = os.path.join(application_dir, folder_path)
                folder_path = "oscarp"
                self.oscarp_path = os.path.join(application_dir, folder_path)
                folder_path = "space4ai-r"
                self.space4air_path = os.path.join(application_dir, folder_path)
                folder_path = "aisprint/deployments/optimal_deployment"
                self.optimal_deployment_path = os.path.join(application_dir, folder_path)

                folder_path = "space4ai-r/common_config"
                target = os.path.join(application_dir, folder_path)
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.copytree(self.common_config_path, target)
                self.common_config_path = target
        else:
            self.error.log("The role of parser's user must be specified, space4ai-d or space4ai-r.")
            sys.exit(1)
        self.get_names_to_code()

    def get_names_to_code(self):
        filename = "component_partitions.yaml"
        filepath = os.path.join(self.component_partitions_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["components"]
        self.names_to_code = {}
        c = 0
        for key, value in components.items():
            # print(key, value)
            self.names_to_code[key] = {}
            c += 1
            h = 0
            for partition in sorted(value["partitions"]):
                # print(partition)
                h += 1
                if partition != "base":
                    s = int(partition.strip("partition").split('_')[0]) + 1
                    # print(s)
                else:
                    s = 1

                self.names_to_code[key][partition] = {
                    "c": "c" + str(c),
                    "s": "s" + str(s),
                    "h": "h" + str(h)
                }

    def is_degraded(self):
        filename = "annotations.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            items = yaml.full_load(file)
        self.degraded = False
        for item in items:
            if "model_performance" in items[item]:
                self.degraded = True
                break
        return self.degraded

