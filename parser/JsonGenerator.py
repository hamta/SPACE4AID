import os
import yaml
import json
import re
import sys
from classes.Logger import Logger
from parser.Parser import Parser


class ParserYamlToJson(Parser):

    def __init__(self, application_dir, who, directory="", log=Logger()):
        super().__init__(application_dir, who, directory, log)

    def get_components(self):
        filename = "component_partitions.yaml"
        filepath = os.path.join( self.component_partitions_path, filename)
        with open(filepath) as file:
            component_partitions = yaml.full_load(file)["components"]

        filename = "application_dag.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            dag = yaml.full_load(file)["System"]["dependencies"]

        data_size_filename = "components_data_size.yaml"
        filepath = os.path.join(self.oscarp_path, data_size_filename)
        with open(filepath) as file:
            data_sizes = yaml.full_load(file)
        components = {}  # returned dict
        number_of_components = len(component_partitions)
        # cycling on numbers so that I know when I reached the end
        for i in range(number_of_components):
            component_name = list(component_partitions.keys())[i]
            partitions = component_partitions[component_name]["partitions"]
            for p in sorted(partitions):
                c, s, h = self.names_to_code[component_name][p].values()
                # base, partition1_1...
                next_component = [item[1] for item in dag if item[0] == component_name]
                next_component_code =[]
                if p == "base":
                    components[c] = {}
                    if len(next_component) > 0:
                        for comp in next_component:
                            next_component_code.append(self.names_to_code[comp]["base"]["c"])
                    '''if i + 1 < number_of_components:
                        next_component = list(component_partitions.keys())[i + 1]
                        next_component, _, _ = self.names_to_code[next_component]["base"].values()
                    else:  # I'm at the end of the line and there is no next
                        next_component = ""'''
                    components[c][s] = {h: {
                        # "memory": memory,
                        "next": next_component_code,
                        "early_exit_probability": 0,  # todo remove hardcode
                        "data_size": [data_sizes[component_name]]  # todo remove hardcode
                    }}
                else:
                    # splitting "partition1_1" in "partition1_" and "1"
                    partition, n = p.split('_')
                    next_component_part = partition + "_" + str(int(n) + 1)
                    if next_component_part not in partitions:
                        if len(next_component) > 0:
                            for comp in next_component:
                                next_component_code.append(self.names_to_code[comp]["base"]["c"])
                        '''if i + 1 < number_of_components:
                            next_component = list(component_partitions.keys())[i + 1]
                            next_component, _, _ = self.names_to_code[next_component]["base"].values()
                        else:  # I'm at the end of the line and there is no next
                            next_component = []'''
                    else:
                        next_component_code.append(self.names_to_code[component_name][next_component_part]["h"])
                    if s not in components[c].keys():
                        components[c][s] = {}
                    next_component_name = component_name + "_" + p
                    components[c][s][h] = {
                        # "memory": memory,
                        "next": next_component_code,
                        "early_exit_probability": 0,  # todo remove hardcode
                        "data_size": [data_sizes[next_component_name]]  # todo remove hardcode
                    }
        return components
     
    def get_components_details(self, name, resource):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            candidate_components = yaml.full_load(file)["Components"]
    
        containers = []
    
        for c in candidate_components:
            component = candidate_components[c]
            if "partition" not in name:
                if component["name"] == name:
                    containers = component["Containers"]
                    break
            else:
                if "partition" in component["name"]:  # no point in checking otherwise
                    # group is the first value, number is the second
                    partition_group_target = re.findall(r'\d+', name)[0]
                    partition_number_target = re.findall(r'\d+', name)[1]
    
                    # component name should have only one number
                    partition_number_component = re.findall(r'\d+', component["name"])[0]
    
                    if partition_number_target == partition_number_component:  # otherwise it's not a match
                        target_name = name.strip(partition_number_target)  # removes last number
                        target_name = target_name.replace(partition_group_target, "A")  # replace first number with letter
    
                        component_name = component["name"].strip(partition_number_component)
    
                        # at this point I have ensured that the names end with "X_", or equivalent
    
                        if target_name[:-2] == component_name[:-2]:
                            containers = component["Containers"]
                            break
    
        for _, container in containers.items():
            if resource in container["candidateExecutionResources"]:
                return container["memorySize"]
    
        return None
      
    def get_resources(self):
        filename = "SPACE4AI-D.yaml"
        filepath = os.path.join( self.space4aid_path, filename)
        with open(filepath) as file:
            space = yaml.full_load(file)
    
        resources = {}
        resources_types = ["EdgeResources", "CloudResources", "FaaSResources"]
    
        for resources_type in resources_types:
            if resources_type in space.keys():
                resources[resources_type] = {}
                layers = space[resources_type]
                for layer_name in layers:
                    resources[resources_type][layer_name] = {}
                    layer = self.get_layer(layer_name)

                    for resource in layer["Resources"]:
                        resource = layer["Resources"][resource]
                        name = resource["name"]
                        if "faas" in name.lower():
                            # get the transitionCost of last FaaS as general transitionCost for json
                            resources[resources_type][layer_name]["transition_cost"] = resource["transitionCost"]

                            resources[resources_type][layer_name][name] = {
                                "description": resource["description"],
                                "cost": resource["cost"],
                                "memory": resource["memorySize"],
                                "idle_time_before_kill": resource["idleTime"]
                            }
                        else:
                            n_cores = resource["processors"]["processor1"]["computingUnits"]

                            resources[resources_type][layer_name][name] = {
                                "description": resource["description"],
                                "number": resource["totalNodes"],
                                "cost": resource["cost"],
                                "memory": resource["memorySize"],
                                "n_cores": n_cores
                            }
    
        return resources
      
    def get_layer(self, target_layer):
        filename = "candidate_resources.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            network_domains = yaml.full_load(file)["System"]["NetworkDomains"]
    
        for domain_id in network_domains.keys():
            layers = network_domains[domain_id]["ComputationalLayers"].keys()
            for layer in layers:
                if layer.lower() == target_layer.lower():
                    return network_domains[domain_id]["ComputationalLayers"][layer]
    
        return None
       
    def get_network(self):
        filename = "candidate_resources.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            network_domains = yaml.full_load(file)["System"]["NetworkDomains"]
    
        network_tech = {}
    
        for domain_id in network_domains.keys():
            access_delay = network_domains[domain_id]["AccessDelay"]
            bandwidth = network_domains[domain_id]["Bandwidth"]
            layers = self.get_domain_layers(domain_id)
    
            network_tech[domain_id] = {
                "computationalLayers": layers,
                "AccessDelay": access_delay,
                "Bandwidth": bandwidth
            }
    
        return network_tech
      
    def get_domain_layers(self, target_domain_id):
        filename = "candidate_resources.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            network_domains = yaml.full_load(file)["System"]["NetworkDomains"]
    
        layers = []
    
        for domain_id in network_domains.keys():
            if target_domain_id == domain_id:
                if "ComputationalLayers" in network_domains[domain_id]:
                    layers += list(network_domains[domain_id]["ComputationalLayers"].keys())
                for subdomain_id in network_domains[domain_id]["subNetworkDomains"]:
                    layers += self.get_domain_layers(subdomain_id)
    
        return layers
       
    def get_compatibility_matrix(self):
        filename = "component_partitions.yaml"
        filepath = os.path.join( self.component_partitions_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["components"]
    
        compatibility_matrix = {}
    
        for component in components:
            for partition in sorted(components[component]["partitions"]):
                if partition == "base":
                    resources = self.get_component_resources(component, "")
                    component_name = component
                    # compatibility_matrix[component][component] = resources
                else:
                    resources = self.get_component_resources(component, partition)
                    # compatibility_matrix[component][component + "_" + partition] = resources
                    component_name = component + "_" + partition
                for resource in resources:
                    memory = self.get_components_details(component_name, resource)
                    c, s, h = self.names_to_code[component][partition].values()
    
                    if c not in compatibility_matrix.keys():
                        compatibility_matrix[c] = {}
    
                    if h not in compatibility_matrix[c].keys():
                        compatibility_matrix[c][h] = []
    
                    compatibility_matrix[c][h].append({
                        "resource": resource,
                        "memory": memory
                    })
    
        # print(compatibility_matrix)
    
        return compatibility_matrix 
    
    def get_component_resources(self, target_component_name, target_partition):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["Components"]
        for component_name in components.keys():
            resources = []
            component = components[component_name]
    
            if target_partition != "":  # if target partition not empty
                if "partition" in component["name"]:
                    partition_group_target = re.findall(r'\d+', target_partition)[0]
                    partition_number_target = re.findall(r'\d+', target_partition)[1]
    
                    partition_number_component = re.findall(r'\d+', component["name"])[0]
                    name = component["name"].split("_partition")[0]
    
                    # print(target_component_name, partition_group_target, partition_number_target)
                    # print(name, " ", partition_number_component)
    
                    if target_component_name == name and partition_number_target == partition_number_component:
                        containers = component["Containers"]
                        for container_name in containers:
                            container = containers[container_name]
                            for r in container["candidateExecutionResources"]:
                                if r not in resources:
                                    resources.append(r)
                        return resources
    
            else:
                if component["name"] == target_component_name:
                    containers = component["Containers"]
                    for container_name in containers:
                        container = containers[container_name]
                        for r in container["candidateExecutionResources"]:
                            if r not in resources:
                                resources.append(r)
                    return resources
       
    def get_component_name(self, component_id):
        filename = "candidate_deployments.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            components = yaml.full_load(file)["Components"]
    
        for component_name in components.keys():
            if components[component_name]["name"] == component_id:
                return components[component_name]["name"]
    
    def get_local_constraints(self):
        filename = "qos_constraints.yaml"
        filepath = os.path.join( self.space4aid_path, filename)
        with open(filepath) as file:
            constraints = yaml.full_load(file)["System"]["local_constraints"]
    
        local_constraints = {}
    
        for constraint in constraints:
            component_id = constraints[constraint]["component_name"]
            threshold = constraints[constraint]["threshold"]
            name = self.get_component_name(component_id)
            name, _, _ = self.names_to_code[name]["base"].values()
            local_constraints[name] = {"local_res_time": threshold}
    
        return local_constraints
    
    def get_global_constraints(self):
        filename = "qos_constraints.yaml"
        filepath = os.path.join(self.space4aid_path, filename)
        with open(filepath) as file:
            constraints = yaml.full_load(file)["System"]["global_constraints"]
    
        global_constraints = {}
    
        for constraint in constraints:
            path_components = constraints[constraint]["path_components"]
            threshold = constraints[constraint]["threshold"]
            components = []
            for component_id in path_components:
                name = self.get_component_name(component_id)
                name, _, _ = self.names_to_code[name]["base"].values()
                components.append(name)
            global_constraints[constraint] = {
                "components": components,
                "global_res_time": threshold
            }
    
        return global_constraints
    
    def get_dag(self):
        filename = "application_dag.yaml"
        filepath = os.path.join( self.common_config_path, filename)
        with open(filepath) as file:
            dependencies = yaml.full_load(file)["System"]["dependencies"]
    
        dag = {}
    
        for dependency in dependencies:
            component_a = dependency[0]
            component_b = dependency[1]
            transition_probability = dependency[2]
    
            component_a, _, _ = self.names_to_code[component_a]["base"].values()
            component_b, _, _ = self.names_to_code[component_b]["base"].values()
    
            if component_a not in dag.keys():
                dag[component_a] = {
                    "next": [component_b],
                    "transition_probability": [transition_probability],
                }
            else:
                dag[component_a]["next"].append(component_b)
                dag[component_a]["transition_probability"].append(transition_probability)
    
        return dag
        
    def get_performance_models(self):
        filename = "performance_models.json"
        filepath = os.path.join( self.oscarp_path, filename)
        with open(filepath) as file:
            data = json.load(file)

        performance = {}
        for component_name, component in data.items():
            for partition_name, partition in component.items():
                print(component_name, partition_name)
                if component_name == partition_name:  # base
                    c, _, h = self.names_to_code[component_name]["base"].values()
                    performance[c] = {}
                    performance[c][h] = partition
                else:
                    #if partition_name in self.names_to_code:
                    partition_name = partition_name.strip(component_name + "_")
                    c, _, h = self.names_to_code[component_name][partition_name].values()
                    performance[c][h] = partition
        return performance

     ## Method to get selected resources to create the candidate resources for SPACE4AI-R (partitioned version)
    def get_selected_res(self):
        filename = "production_deployment.yaml"
        filepath = os.path.join(self.optimal_deployment_path, filename)
        with open(filepath) as file:
            selected_NDs = yaml.full_load(file)["System"]["Resources"]["NetworkDomains"]

        filename = "candidate_resources.yaml"
        filepath = os.path.join(self.common_config_path, filename)
        with open(filepath) as file:
            system = yaml.full_load(file)["System"]
        candidate_res = {"System": {"name": system["name"], "NetworkDomains": selected_NDs}}
        return candidate_res

    def make_system_file(self):
        if self.who == "SPACE4AI-R" and not self.degraded:
            candidate_res = self.get_selected_res()
            filename = "candidate_resources.yaml"
            filepath = os.path.join(self.common_config_path, filename)
            with open(filepath, "w") as file:
                json.dump(candidate_res, file, sort_keys=False)
        system_file = self.get_resources()
        system_file["Components"] = self.get_components()
        system_file["NetworkTechnology"] = self.get_network()
        system_file["CompatibilityMatrix"] = self.get_compatibility_matrix()
        system_file["LocalConstraints"] = self.get_local_constraints()
        system_file["GlobalConstraints"] = self.get_global_constraints()
        system_file["DirectedAcyclicGraph"] = self.get_dag()
        system_file["Performance"] = self.get_performance_models()
        system_file["Lambda"] = 0.25
        system_file["Time"] = 1

        filename = "SystemFile.json"
        if self.who == "SPACE4AI-R":
            filepath = os.path.join(self.space4air_path, filename)
        else:
            filepath = os.path.join( self.space4aid_path, filename)
        with open(filepath, "w") as file:
            json.dump(system_file, file, indent=4)

        return filepath

    def make_input_json(self):
        filename = "SPACE4AI-D.yaml"
        filepath = os.path.join( self.space4aid_path, filename)

        with open(filepath) as file:
            content = yaml.load(file, Loader=yaml.Loader)
            methods = content["Methods"]
            seed = content["Seed"]
            verbose_level = content["VerboseLevel"]

        filename = "Input.json"
        filepath = os.path.join( self.space4aid_path, filename)

        with open(filepath, "w") as file:
            json.dump({"Methods": methods, "Seed": seed, "VerboseLevel": verbose_level}, file, indent=4)
        return filepath
