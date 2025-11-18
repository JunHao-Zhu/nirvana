from nirvana.lineage.abstractions import LineageNode


def rewire_nodes(start_node: LineageNode, node_list: list[LineageNode]):
    last_node = start_node
    for node in node_list:
        node.node_fields.left_input_fields = last_node.node_fields.output_fields
        node.node_fields.output_fields = list(
            set(node.node_fields.left_input_fields + node.operator.generated_fields)
        )
        node.set_left_child(last_node)
        last_node = node
    return last_node


class MapPullup:
    @classmethod
    def check_pattern(cls, dependencies: list[str], generated_fields: list[str]) -> bool:
        return all([field not in generated_fields for field in dependencies])

    @classmethod
    def split_maps_into_pullup_and_stay(cls, node: LineageNode, maps: list[LineageNode]) -> tuple[list, list]:
        maps_to_pullup, maps_to_stay = [], []
        for map in maps:
            dependencies = node.operator.dependencies
            map_generated_fields = map.operator.generated_fields
            can_pullup = cls.check_pattern(dependencies, map_generated_fields)
            if can_pullup:
                maps_to_pullup.append(map)
            else:
                maps_to_stay.append(map)
        return maps_to_pullup, maps_to_stay
    
    @classmethod
    def pull_up(cls, node: LineageNode) -> tuple[LineageNode, list[LineageNode]]:
        if node.op_name == "scan":
            return node, []

        elif node.op_name in ["filter", "rank"]:
            child_node, pullup_maps = cls.pull_up(node.left_child)
            maps_to_pullup, maps_to_stay = cls.split_maps_into_pullup_and_stay(node, pullup_maps)
            last_node = rewire_nodes(child_node, maps_to_stay)
            
            node.node_fields.left_input_fields = last_node.node_fields.output_fields
            node.node_fields.output_fields = node.node_fields.left_input_fields
            node.set_left_child(last_node)

            last_node = rewire_nodes(node, maps_to_pullup)
            return node, maps_to_pullup

        elif node.op_name == "join":
            left_child_node, left_pullup_maps = cls.pull_up(node.left_child)
            right_child_node, right_pullup_maps = cls.pull_up(node.right_child)

            left_maps_to_pullup, left_maps_to_stay = cls.split_maps_into_pullup_and_stay(node, left_pullup_maps)
            right_maps_to_pullup, right_maps_to_stay = cls.split_maps_into_pullup_and_stay(node, right_pullup_maps)

            left_last_node = rewire_nodes(left_child_node, left_maps_to_stay)
            right_last_node = rewire_nodes(right_child_node, right_maps_to_stay)
            
            node.node_fields.left_input_fields = left_last_node.node_fields.output_fields
            node.node_fields.right_input_fields = right_last_node.node_fields.output_fields

            maps_to_pullup = left_maps_to_pullup + right_maps_to_pullup
            last_node = rewire_nodes(node, maps_to_pullup)
            return node, maps_to_pullup

        elif node.op_name == "map":
            child_node, pullup_maps = cls.pull_up(node.left_child)
            maps_to_pullup, maps_to_stay = cls.split_maps_into_pullup_and_stay(node, pullup_maps)
            pullup_maps = maps_to_stay + [node] + maps_to_pullup
            last_node = rewire_nodes(child_node, pullup_maps)
            return child_node, pullup_maps

        elif node.op_name == "reduce":
            child_node, pullup_maps = cls.pull_up(node.left_child)
            last_node = rewire_nodes(child_node, pullup_maps)
            node.node_fields.left_input_fields = last_node.node_fields.output_fields
            node.set_left_child(last_node)
            return node, pullup_maps

    @classmethod
    def transform(cls, node: LineageNode):
        node, _ = cls.pull_up(node)
        return node
