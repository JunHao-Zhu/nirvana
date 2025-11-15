from nirvana.lineage.abstractions import LineageNode


# class MapPullup:
#     # TODO: fix bugs when pulling up multiple maps in a branch
    
#     @classmethod
#     def check_pattern(cls, dependencies: list[str], generated_fields: list[str]) -> bool:
#         return all([field not in generated_fields for field in dependencies])

#     @classmethod
#     def transform(cls, node: LineageNode) -> LineageNode:
#         if node.op_name == "join":
#             # First, transform both branches below the join
#             node.set_left_parent(cls.transform(node.left_parent))
#             node.set_right_parent(cls.transform(node.right_parent))

#             # Attempt to pull up a map from the left branch if safe
#             left_parent = node.left_parent
#             pulled_maps = []
#             if left_parent.op_name == "map":
#                 generated_fields = left_parent.operator.generated_fields
#                 left_dependencies = node.operator.left_on
#                 if cls.check_pattern(left_dependencies, generated_fields):
#                     # If the join does not reference the produced column from the left side, it is safe to pull up
#                     new_map = LineageNode(
#                         op_name="map",
#                         op_kwargs=left_parent.operator.op_kwargs,
#                         node_fields=left_parent.node_fields.model_dump(),
#                     )

#                     # Update join's left input fields to what the map previously consumed
#                     node.node_fields.left_input_fields = left_parent.node_fields.output_fields

#                     # Establish the join output fields (union of both sides)
#                     join_output_fields = list(
#                         set(node.node_fields.left_input_fields + node.node_fields.right_input_fields)
#                     )
#                     node.node_fields.output_fields = join_output_fields

#                     # Rewire: join connects to the map's previous parent; map connects after join
#                     node.set_left_parent(left_parent.left_parent)
#                     del left_parent
#                     pulled_maps.append(new_map)

#             # Attempt to pull up a map from the right branch if safe
#             right_parent = node.right_parent
#             if right_parent.op_name == "map":
#                 generated_fields = right_parent.operator.generated_fields
#                 right_dependencies = node.operator.right_on
#                 if cls.check_pattern(right_dependencies, generated_fields):
#                     # If the join does not reference the produced column from the right side, it is safe to pull up
#                     new_map = LineageNode(
#                         op_name="map",
#                         op_kwargs=right_parent.operator.op_kwargs,
#                         node_fields=right_parent.node_fields.model_dump(),
#                     )

#                     # Update join's right input fields to what the map previously consumed
#                     node.node_fields.right_input_fields = right_parent.node_fields.left_input_fields

#                     # Establish the join output fields (union of both sides)
#                     join_output_fields = list(
#                         set(node.node_fields.left_input_fields + node.node_fields.right_input_fields)
#                     )
#                     node.node_fields.output_fields = join_output_fields

#                     # Rewire: join connects to the map's previous parent; map connects after join
#                     node.set_right_parent(right_parent.left_parent)
#                     del right_parent
#                     pulled_maps.append(new_map)
#             prev_node = node
#             for map in pulled_maps:
#                 # The pulled-up map now consumes the previous node's (new map's or join's) outputs and adds its produced column
#                 map.node_fields.left_input_fields = prev_node.node_fields.output_fields
#                 map.node_fields.output_fields = list(
#                     set(map.node_fields.left_input_fields + map.operator.generated_fields)
#                 )
#                 map.set_left_parent(prev_node)
#                 prev_node = map
#             return prev_node

#         elif node.op_name in ["map", "filter"]:
#             # First, transform the subtree below
#             node.set_left_parent(cls.transform(node.left_parent))

#             parent = node.left_parent
#             # If the parent is a map and the current node does not reference the map's produced column,
#             # pull the map up (i.e., after the current node)
#             if parent.op_name == "map":
#                 generated_fields = parent.operator.generated_fields
#                 dependencies = node.operator.dependencies
#                 # Only safe to pull up when the child doesn't reference the produced column
#                 if cls.check_pattern(dependencies, generated_fields):
#                     new_map = LineageNode(
#                         op_name="map",
#                         op_kwargs=parent.operator.op_kwargs,
#                         node_fields=parent.node_fields.model_dump(),
#                     )

#                     # Update metadata to reflect the new ordering
#                     # Child now receives what the map used to receive
#                     node.node_fields.left_input_fields = parent.node_fields.left_input_fields
#                     node.node_fields.output_fields = list(
#                         set(node.node_fields.left_input_fields + node.operator.generated_fields)
#                     )

#                     # The pulled-up map now consumes the child's outputs
#                     new_map.node_fields.left_input_fields = node.node_fields.output_fields
#                     new_map.node_fields.output_fields = list(
#                         set(new_map.node_fields.left_input_fields + new_map.operator.generated_fields)
#                     )

#                     # Rewire: node connects to map's previous parent; map connects after node
#                     node.set_left_parent(parent.left_parent)
#                     new_map.set_left_parent(node)
#                     del parent
#                     return new_map

#             return node

#         elif node.op_name == "reduce":
#             node.set_left_parent(cls.transform(node.left_parent))
#             return node

#         else:
#             return node


def rewire_nodes(start_node: LineageNode, node_list: list[LineageNode]):
    last_node = start_node
    for node in node_list:
        node.node_fields.left_input_fields = last_node.node_fields.output_fields
        node.node_fields.output_fields = list(
            set(node.node_fields.left_input_fields + node.operator.generated_fields)
        )
        node.set_left_parent(last_node)
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
            parent_node, pullup_maps = cls.pull_up(node.left_parent)
            maps_to_pullup, maps_to_stay = cls.split_maps_into_pullup_and_stay(node, pullup_maps)
            last_node = rewire_nodes(parent_node, maps_to_stay)
            
            node.node_fields.left_input_fields = last_node.node_fields.output_fields
            node.node_fields.output_fields = node.node_fields.left_input_fields
            node.set_left_parent(last_node)

            last_node = rewire_nodes(node, maps_to_pullup)
            return node, maps_to_pullup

        elif node.op_name == "join":
            left_parent_node, left_pullup_maps = cls.pull_up(node.left_parent)
            right_parent_node, right_pullup_maps = cls.pull_up(node.right_parent)

            left_maps_to_pullup, left_maps_to_stay = cls.split_maps_into_pullup_and_stay(node, left_pullup_maps)
            right_maps_to_pullup, right_maps_to_stay = cls.split_maps_into_pullup_and_stay(node, right_pullup_maps)

            left_last_node = rewire_nodes(left_parent_node, left_maps_to_stay)
            right_last_node = rewire_nodes(right_parent_node, right_maps_to_stay)
            
            node.node_fields.left_input_fields = left_last_node.node_fields.output_fields
            node.node_fields.right_input_fields = right_last_node.node_fields.output_fields

            maps_to_pullup = left_maps_to_pullup + right_maps_to_pullup
            last_node = rewire_nodes(node, maps_to_pullup)
            return node, maps_to_pullup

        elif node.op_name == "map":
            parent_node, pullup_maps = cls.pull_up(node.left_parent)
            maps_to_pullup, maps_to_stay = cls.split_maps_into_pullup_and_stay(node, pullup_maps)
            pullup_maps = maps_to_stay + [node] + maps_to_pullup
            last_node = rewire_nodes(parent_node, pullup_maps)
            return parent_node, pullup_maps

        elif node.op_name == "reduce":
            parent_node, pullup_maps = cls.pull_up(node.left_parent)
            last_node = rewire_nodes(parent_node, pullup_maps)
            node.node_fields.left_input_fields = last_node.node_fields.output_fields
            node.set_left_parent(last_node)
            return node, pullup_maps

    @classmethod
    def transform(cls, node: LineageNode):
        node, _ = cls.pull_up(node)
        return node
