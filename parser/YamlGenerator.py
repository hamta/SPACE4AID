import copy
import os
import yaml
import json
from classes.Logger import Logger
from parser.Parser import Parser
import sys
import shutil


class ParserJsonToYaml(Parser):

    def __init__(self, application_dir, who, directory="", log=Logger()):
        super().__init__(application_dir, who, directory, log)

    def parse_output_json(self):
        filename = "Output.json"
        filepath = os.path.join(self.space4aid_path, filename)
        with open(filepath) as file:
            data = json.load(file)

        c_keys = data["components"].keys()
        # print(c_keys)
        feasibility = data["feasible"]
        components_values = []

        for c in c_keys:
            s_keys = data["components"][c].keys()
            # print(s_keys)
            for s in s_keys:
                if type(data["components"][c][s]) is dict:
                    h_keys = data["components"][c][s].keys()
                    # print(h_keys)
                    for h in h_keys:
                        computational_layer = data["components"][c][s][h]
                        components_values.append(((c, s, h), computational_layer))

        # print("Components values: ", components_values)
        return feasibility, components_values

    def find_right_components(self, components_values):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            deployments = yaml.full_load(file)

        d_keys = deployments["Components"].keys()
        d_values = deployments["Components"].values()

        components = {}

        for component_values in components_values:
            codes = component_values[0]
            c = codes[0]
            s = codes[1]
            h = codes[2]
            # c = c.replace("c", "component")  # component1
            # h = int(h.replace("h", ""))  # 1

            for comp_name, value in self.names_to_code.items():
                if c == value["base"]["c"]:
                    for key, val in deployments["Components"].items():
                        if val["name"] == comp_name:
                            target = key
                            break
                    #target = c.replace("c", "component")  # component1

                    if s == "s1":  # base partition
                        break
                    else:
                        for partition, code in value.items():
                            if s == code["s"] and h == code["h"]:
                                partition = partition[:-3] + "X" + partition[-2:]
                                target = target + "_" + partition
                                break

            for d in d_keys:  # [ component1, component1_partitionX_1, ... ]
                comp_name = deployments["Components"][d]["name"]
                # print(deployments["Components"][d]["name"])
                # print(h)
                if d == target:
                    components[target] = self.update_component(deployments["Components"][d], component_values[1], comp_name)
                    break

        # components = {"Components": components}

        return components

    def update_component(self, old_component, layer_info, name):
        # copies component body from candidate_deployments.yaml
        new_component = old_component

        # fix name
        new_component["name"] = name


        # rename layer key and puts in the correct value
        computational_layer = list(layer_info.keys())[0]
        exec_layer = int(computational_layer.replace("computationalLayer", ""))
        new_component.pop("candidateExecutionLayers")
        new_component["executionLayer"] = exec_layer

        # choose right container
        containers = old_component["Containers"]

        # picks the correct resource from Output.json
        chosen_resource = list(layer_info[computational_layer].keys())[0]

        # if there's only one container, job done
        if len(containers.keys()) == 1:
            key = list(containers.keys())[0]

        # else find the right one
        else:
            for key in containers.keys():

                # if there's a match we found our container
                # no error control if no match found, assuming not possible
                if chosen_resource in containers[key]["candidateExecutionResources"]:
                    break

        # select correct container and rename key
        chosen_container = containers[key]
        chosen_container.pop("candidateExecutionResources")
        chosen_container["selectedExecutionResource"] = chosen_resource

        # save correct container in component
        new_component["Containers"] = {"container1": chosen_container}

        return new_component

    def find_right_resources(self, layers):
        filename = "candidate_resources.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            candidate_resources = yaml.full_load(file)["System"]

        net_domains = candidate_resources["NetworkDomains"].keys()

        comp_layer_resource_tuple = []

        # this associates the valid components from output.json to the one from candidate_resources.yaml
        for layer in layers:
            computational_layer = list(layer[1].keys())[0]
            resource = list(layer[1][computational_layer].keys())[0]

            comp_layer_resource_tuple.append((computational_layer, resource))

        # print(comp_layer_resource_tuple)
        to_pop = []

        for n in net_domains:
            n_keys = candidate_resources["NetworkDomains"][n].keys()
            if "ComputationalLayers" in n_keys:
                computational_layers = candidate_resources["NetworkDomains"][n]["ComputationalLayers"]
                cl_keys = computational_layers.keys()
                # print(cl_keys)
                for cl in cl_keys:
                    resources_keys = computational_layers[cl]["Resources"].keys()
                    for r in resources_keys:
                        resource = computational_layers[cl]["Resources"][r]["name"]
                        # print((cl, resource))
                        # if not a winner component, pop
                        if (cl, resource) not in comp_layer_resource_tuple:
                            # print("popped")
                            to_pop.append([n, cl, r])
                        # otherwise, make sure that it has the correct number of nodes
                        else:
                            index = comp_layer_resource_tuple.index((cl, resource))
                            if "number" in layers[index][1][cl][resource].keys():
                                correct_node_number = layers[index][1][cl][resource]["number"]
                                computational_layers[cl]["Resources"][r]["totalNodes"] = correct_node_number

        for array in to_pop:
            n = array[0]
            cl = array[1]
            r = array[2]

            candidate_resources["NetworkDomains"][n]["ComputationalLayers"][cl]["Resources"].pop(r)

        return self.fix_resources(candidate_resources)

    def fix_resources(self, candidate_resources):
        # If layer has no more resources, delete layer #
        # otherwise fix resources number to fill gaps
        to_pop = []
        net_domains = candidate_resources["NetworkDomains"].keys()
        for n in net_domains:
            n_keys = candidate_resources["NetworkDomains"][n].keys()
            if "ComputationalLayers" in n_keys:
                computational_layers = candidate_resources["NetworkDomains"][n]["ComputationalLayers"]
                cl_keys = computational_layers.keys()
                # print(cl_keys)
                for cl in cl_keys:
                    resources = computational_layers[cl]["Resources"]
                    resources_keys = list(resources.keys())
                    if len(resources_keys) == 0:
                        to_pop.append([n, cl])
                    else:
                        for i in range(0, len(resources_keys)):
                            if resources_keys[i] != "resource" + str(i + 1):
                                resources["resource" + str(i + 1)] = resources.pop(resources_keys[i])
                                # print(resources_keys[i])

        for array in to_pop:
            n = array[0]
            cl = array[1]
            candidate_resources["NetworkDomains"][n]["ComputationalLayers"].pop(cl)

        # if net domain has no more layers, delete domain
        to_pop = []
        net_domains = candidate_resources["NetworkDomains"].keys()
        no_comp_layers_domains = []
        for n in net_domains:
            n_keys = candidate_resources["NetworkDomains"][n].keys()
            if "ComputationalLayers" in n_keys:
                computational_layers = candidate_resources["NetworkDomains"][n]["ComputationalLayers"]
                cl_keys = computational_layers.keys()
                if len(cl_keys) == 0:
                    to_pop.append(n)
            else:
                no_comp_layers_domains.append(n)

        for n in to_pop:
            candidate_resources["NetworkDomains"].pop(n)

        # at this point all NDs have either valid comp_layers or none
        # if they have no comp_layers and no valid subdomains, delete domain
        net_domains = set(candidate_resources["NetworkDomains"].keys())
        net_domains -= set(no_comp_layers_domains)

        to_pop = []
        for n in no_comp_layers_domains:
            sub_nd = candidate_resources["NetworkDomains"][n]["subNetworkDomains"]
            valid_sub_nds = []
            for s in sub_nd:
                if s in net_domains:
                    valid_sub_nds.append(s)

            # if not empty
            if valid_sub_nds:
                candidate_resources["NetworkDomains"][n]["subNetworkDomains"] = valid_sub_nds
            else:
                to_pop.append(n)

        for n in to_pop:
            candidate_resources["NetworkDomains"].pop(n)

        return candidate_resources

    def make_output_yaml(self, feasible, components, resources):
        output = {"System": {
            "Components": components,
            "Resources": resources,
            "Feasible": feasible
        }}

        filename = "production_deployment.yaml"
        if self.who == "SPACE4AI-R":
            filepath = os.path.join(self.space4air_path, filename)
        else:
            filepath = os.path.join(self.optimal_deployment_path, filename)

        with open(filepath, "w") as file:
            yaml.dump(output, file, sort_keys=False)

    ## Method to return the candidate resources for a specific component
    def get_candidate_resources_component(self, target_component_name):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["Components"]

        for component_name in components.keys():
            resources = []
            component = components[component_name]
            if component["name"] == target_component_name:
                containers = component["Containers"]
                for container_name in containers:
                    container = containers[container_name]
                    for r in container["candidateExecutionResources"]:
                        if r not in resources:
                            resources.append(r)
                    return resources

    ## Method to return the list of all containers for a specific component
    def get_candidate_resources_container(self, target_component_name, container):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["Components"]
        resources = []
        for component_name in components.keys():
            component = components[component_name]
            if component["name"] == target_component_name:
                containers = component["Containers"]
                for container_name in containers:
                    if container_name == container:
                        resources.extend(containers[container_name]["candidateExecutionResources"])
                return resources

   ## Method to return the list of selected resources in production_deployment.yaml
    def get_selected_resources(self):
        filename = "production_deployment.yaml"
        filepath = os.path.join(self.optimal_deployment_path, filename)
        with open(filepath) as file:
            NDs = yaml.full_load(file)["System"]["Resources"]["NetworkDomains"]
        selected_resources = []
        for ND in NDs.keys():
            CLs = NDs[ND]["ComputationalLayers"]
            for CL in CLs.keys():
                resources = CLs[CL]["Resources"]
                for res in resources.keys():
                    selected_resources.append(resources[res]["name"])
        # get all FaaS resources
        filename = "candidate_resources.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            NDs = yaml.full_load(file)["System"]["NetworkDomains"]
        for ND in NDs.keys():
            CLs = NDs[ND]["ComputationalLayers"]
            for CL in CLs.keys():
                if "faas" in CLs[CL]["name"].lower():
                    resources = CLs[CL]["Resources"]
                    for res in resources.keys():
                        selected_resources.append(resources[res]["name"])
        return selected_resources

    ## Method to drope partitions if it is degraded version
    def drop_partitions_from_candidate_dep(self):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            candidate_components = yaml.full_load(file)["Components"]
            candidate_components_copy = copy.deepcopy(candidate_components)
        for component in candidate_components.keys():
            if "partition" in component.lower() or "partition" in candidate_components[component]["name"].lower():
                candidate_components_copy.pop(component, None)
        candidates = {"Components": candidate_components_copy}
        return candidates

    ## Method to replace the name of degraded component with the base one in qos_constraints
    def replace_comp_name_QoS(self, comp_name, new_comp_name, qos_constraints):
        for key, value in qos_constraints.items():
            if key.lower() == "local_constraints":  # found value
               for k, v in value.items():
                   if v["component_name"] == comp_name:
                      qos_constraints["local_constraints"][k]["component_name"] = new_comp_name
            if key.lower() == "global_constraints":
                for k, v in value.items():
                    if comp_name in v["path_components"]:
                        qos_constraints["global_constraints"][k]["path_components"] = list(map(lambda item: item.replace(comp_name, new_comp_name), v["path_components"]))
        return qos_constraints

    ## Method to get selected resources and with all FaaS resources to create the candidate resources for SPACE4AI-R (degraded version)
    def get_selected_res_and_FaaS(self):
        filename = "production_deployment.yaml"
        filepath = os.path.join(self.optimal_deployment_path, filename)
        with open(filepath) as file:
            selected_NDs = yaml.full_load(file)["System"]["Resources"]["NetworkDomains"]

        filename = "candidate_resources.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            system = yaml.full_load(file)["System"]
        NDs = system["NetworkDomains"]
        FaaS_ND_value = None
        for ND in NDs:
            CLs = NDs[ND]["ComputationalLayers"]
            for CL in CLs:
                if "faas" in CLs[CL]["name"].lower():
                    FaaS_cl_key = CL
                    FaaS_cl_value = CLs[CL]
                    FaaS_ND_key = ND
                    FaaS_ND_value = NDs[ND]
        if FaaS_ND_value is not None:
            FaaS_is_selected = False
            for ND in selected_NDs:
                if selected_NDs[ND]["name"] == FaaS_ND_value["name"]:
                    CLs = selected_NDs[ND]["ComputationalLayers"]
                    for CL in CLs:
                        if FaaS_cl_value["name"] == CLs[CL]["name"]:
                            FaaS_is_selected = True
                            selected_NDs[ND]["ComputationalLayers"][CL] = FaaS_cl_value
                    if not FaaS_is_selected:
                        FaaS_is_selected = True
                        selected_NDs[ND]["ComputationalLayers"][FaaS_cl_key] = FaaS_cl_value
            if not FaaS_is_selected:
                selected_NDs[FaaS_ND_key] = FaaS_ND_value
        candidate_res = {"System": {"name": system["name"], "NetworkDomains": selected_NDs}}
        return candidate_res

    ## Method to create deployment folders for SPACE4AI-R includes all required yaml files for degraded version
    def prepare_folders_for_space4air(self):
        filename = "application_dag.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            application_dag = yaml.full_load(file)
        filename = "deployments_performance.yaml"
        filepath = os.path.join(self.space4air_path, filename)
        with open(filepath) as file:
            system = yaml.full_load(file)["System"]
        metric_thr = system["metric_thr"]
        sorted_deps = system["sorted_deployments_performance"]
        selected_res = set(self.get_selected_resources())
        # loop over all degraded deployments to create one folder for each, includes all required yaml
        for dep in sorted_deps:
            dep_name = list(dep.keys())[0]
            if dep[dep_name]["metric_value"] >= metric_thr:
                dep_dir_path = os.path.join(self.space4air_path, dep_name)
                if os.path.exists(dep_dir_path):
                    shutil.rmtree(dep_dir_path)
                os.mkdir(dep_dir_path)
                # create component_partitions.yaml with corresponding degraded component
                filename = "component_partitions.yaml"
                comps = {}
                for comp in dep[dep_name]["components"]:
                    comps[comp] = {"partitions": ["base"]}
                components = {}
                components["components"] = comps
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    yaml.dump(components, file, sort_keys=False)

                # create application_dag.yaml with corresponding degraded component
                filename = "application_dag.yaml"
                application_dag_dep = copy.deepcopy(application_dag)
                if "alternative_components" in application_dag_dep["System"]:
                    application_dag_dep["System"].pop("alternative_components")
                application_dag_dep["System"]["components"] = dep[dep_name]["components"]
                application_dag_dep["System"]["dependencies"] = dep[dep_name]["dependencies"]
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    yaml.dump(application_dag_dep, file, sort_keys=False)

                candidate_components = self.drop_partitions_from_candidate_dep()

                filename = "qos_constraints.yaml"
                filepath = os.path.join(self.space4aid_path, filename)
                with open(filepath) as file:
                    qos_constraints = yaml.full_load(file)["System"]

                # Use performance_models_degraded.json to create proper performance_models.json
                filename = "performance_models_degraded.json"
                filepath = os.path.join(self.oscarp_path, filename)
                with open(filepath) as file:
                    performance_model_degraded = json.load(file)
                performance_model = {}
                comps_name = dep[dep_name]["components"]
                components = {}
                for comp_name in comps_name:
                    degraded = False
                    if comp_name not in application_dag["System"]["components"]:
                        degraded = True
                        for k, v in application_dag["System"]["alternative_components"].items():
                            if comp_name in v:
                                # create qos_constraints dictionary with corresponding degraded component
                                qos_constraints = self.replace_comp_name_QoS(k, comp_name, qos_constraints)
                                degraded_comp_name = comp_name
                                comp_name = k
                                if degraded_comp_name in performance_model_degraded:
                                    performance_model[degraded_comp_name] = {degraded_comp_name: performance_model_degraded[degraded_comp_name]}
                                else:
                                    self.error.log("ML model dose not exist for {}".format(degraded_comp_name))
                                    sys.exit(1)
                                break
                    else:
                        if comp_name in performance_model_degraded:
                            performance_model[comp_name] = {comp_name: performance_model_degraded[comp_name]}
                        else:
                            self.error.log("ML model dose not exist for {}".format(comp_name))
                            sys.exit(1)
                    candidate_res_design = set(self.get_candidate_resources_component(comp_name))
                    candidate_res_runtime = set(candidate_res_design.intersection(selected_res))
                    # create candidate_components dictionary with corresponding degraded component
                    for key, value in candidate_components["Components"].items():
                        if value["name"] == comp_name:
                            containers = value["Containers"]
                            for container_name in containers:
                                candidate_res_container = set(self.get_candidate_resources_container(comp_name, container_name))
                                candidate_res_runtime_container = list(candidate_res_runtime.intersection(candidate_res_container))
                                candidate_components["Components"][key]["Containers"][container_name]["candidateExecutionResources"] = candidate_res_runtime_container
                                if degraded:
                                    candidate_components["Components"][key]["name"] = degraded_comp_name
                            break
                filename = "qos_constraints.yaml"
                qos_constraints = {"System": qos_constraints}
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    yaml.dump(qos_constraints, file, sort_keys=False)

                filename = "candidate_deployments.yaml"
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    yaml.dump(candidate_components, file, sort_keys=False)

                selected_resources = self.get_selected_res_and_FaaS()
                filename = "candidate_resources.yaml"
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    yaml.dump(selected_resources, file, sort_keys=False)

                filename = "performance_models.json"
                filepath = os.path.join(dep_dir_path, filename)
                with open(filepath, "w") as file:
                    json.dump(performance_model, file, sort_keys=False)

                filename = "components_data_size.yaml"
                original = os.path.join(self.oscarp_path, filename)
                target = os.path.join(dep_dir_path, filename)
                shutil.copy(original, target)

                filename = "SPACE4AI-D.yaml"
                original = os.path.join(self.space4aid_path, filename)
                target = os.path.join(dep_dir_path, filename)
                shutil.copy(original, target)

    def main_function(self):
        # extracts useful info from output_json
        feasible, component_values = self.parse_output_json()
        # picks the correct components
        final_components = self.find_right_components(component_values)
        # picks the correct resources
        final_resources = self.find_right_resources(component_values)
        # puts them together in the output.yaml file
        self.make_output_yaml(feasible, final_components, final_resources)

        if self.who == "SPACE4AI-D" and self.is_degraded():
            self.prepare_folders_for_space4air()


