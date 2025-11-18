from nirvana.lineage.abstractions import LineageNode


def rewire_nodes(start_node: LineageNode, node_list: list[LineageNode]):
    last_node = start_node
    for node in node_list:
        node.node_fields.left_input_fields = last_node.node_fields.output_fields
        node.node_fields.output_fields = node.node_fields.left_input_fields
        node.set_left_child(last_node)
        last_node = node
    return last_node


class FilterPullup:
    @classmethod
    def pull_up(cls, node: LineageNode) -> tuple[LineageNode, list[LineageNode]]:
        if node.op_name == "scan":
            return node, []

        elif node.op_name == "filter":
            child_node, pullup_filters = cls.pull_up(node.left_child)
            pullup_filters.append(node)
            last_node = rewire_nodes(child_node, pullup_filters)
            return child_node, pullup_filters

        elif node.op_name == "join":
            left_child_node, left_pullup_filters = cls.pull_up(node.left_child)
            right_child_node, right_pullup_filters = cls.pull_up(node.right_child)

            node.set_left_child(left_child_node)
            node.set_right_child(right_child_node)
            pullup_filters = left_pullup_filters + right_pullup_filters
            last_node = rewire_nodes(node, pullup_filters)
            return node, pullup_filters

        elif node.op_name in ["map", "rank"]:
            child_node, pullup_filters = cls.pull_up(node.left_child)
            # Update operator's left input fields to what the child node previously produced
            node.node_fields.left_input_fields = child_node.node_fields.output_fields
            node.node_fields.output_fields = list(
                set(node.node_fields.left_input_fields + node.operator.generated_fields)
            )
            node.set_left_child(child_node)
            last_node = rewire_nodes(node, pullup_filters)
            return node, pullup_filters

        elif node.op_name == "reduce":
            child_node, pullup_filters = cls.pull_up(node.left_child)
            last_node = rewire_nodes(child_node, pullup_filters)
            node.node_fields.left_input_fields = last_node.node_fields.output_fields
            node.set_left_child(last_node)
            return node, pullup_filters
    
    @classmethod
    def transform(cls, node: LineageNode):
        node, _ = cls.pull_up(node)
        return node
