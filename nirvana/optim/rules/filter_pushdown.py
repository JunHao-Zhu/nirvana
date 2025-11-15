from nirvana.lineage.abstractions import LineageNode


class FilterPushdown:
    @classmethod
    def check_pattern(cls, dependencies: list[str], existing_fields: list[str]) -> bool:
        return all([field in existing_fields for field in dependencies])

    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name == "filter":
            last_node = cls.transform(node.left_parent)
            input_columns = node.operator.input_columns

            if last_node.op_name == "join":
                left_fields = last_node.node_fields.left_input_fields
                right_fields = last_node.node_fields.right_input_fields
                # push filter into the left sub-lineage
                pushdown_flag = False
                if cls.check_pattern(input_columns, left_fields):
                    new_node = LineageNode(op_name="filter", op_kwargs=node.operator.op_kwargs, node_fields=node.node_fields.model_dump())
                    # swap info (eg fields) of current op (eg, filter) and its predecessor (ie join)
                    new_node.node_fields.output_fields = new_node.node_fields.left_input_fields = last_node.node_fields.left_input_fields
                    # rewire edges between current op and its predecessor and rewrite sub-lineage over pushdowned filter
                    new_node.set_left_parent(last_node.left_parent)
                    last_node.set_left_parent(cls.transform(new_node))
                    pushdown_flag = True
                # push filter into the right sub-lineage
                if cls.check_pattern(input_columns, right_fields):
                    new_node = LineageNode(op_name="filter", op_kwargs=node.operator.op_kwargs, node_fields=node.node_fields.model_dump())
                    # swap info (eg fields) of current op, filter, and its predecessor (ie join)
                    new_node.node_fields.output_fields = new_node.node_fields.left_input_fields = last_node.node_fields.right_input_fields
                    # rewire edges between current op and its predecessor and rewrite sub-lineage over pushdowned filter
                    new_node.set_left_parent(last_node.right_parent)
                    last_node.set_right_parent(cls.transform(new_node))
                    pushdown_flag = True
                if pushdown_flag:
                    del node
                    return last_node
                else:
                    node.set_left_parent(last_node)
                    return node
            
            elif last_node.op_name in ["map", "filter", "rank"]:
                fields = last_node.node_fields.left_input_fields
                if cls.check_pattern(input_columns, fields):
                    # swap info (eg fields) of current op, filter, and its predecessor (eg map)
                    node.node_fields.output_fields = node.node_fields.left_input_fields = last_node.node_fields.left_input_fields
                    # rewire edges around current op and its predecessor, and rewrite sub-lineage over pushdowned filter
                    node.set_left_parent(last_node.left_parent)
                    last_node.set_left_parent(cls.transform(node))
                    return last_node
                else:
                    node.set_left_parent(last_node)
                    return node
                
            else:
                node.set_left_parent(last_node)
                return node
                
        elif node.op_name == "join":
            node.set_left_parent(cls.transform(node.left_parent))
            node.set_right_parent(cls.transform(node.right_parent))
            return node
        
        elif node.op_name in ["map", "rank", "reduce"]:
            node.set_left_parent(cls.transform(node.left_parent))
            return node
        
        else:
            return node
