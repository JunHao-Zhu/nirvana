from nirvana.lineage.abstractions import LineageNode


class FilterPushdown:
    @classmethod
    def rewrite_plan(cls, node: LineageNode) -> LineageNode:
        if node.op_name == "filter":
            last_node = node.left_parent
            last_node = cls.rewrite_plan(last_node)
            input_column = node.op_metadata["input_column"]

            if last_node.op_name == "join":
                left_fields = last_node.data_metadata["input_left_fields"]
                right_fields = last_node.data_metadata["input_right_fields"]
                # push filter into the left sub-lineage
                pushdown_flag = False
                if input_column in left_fields:
                    new_node = LineageNode(op_name="filter", op_metadata=node.op_metadata, data_metadata=node.data_metadata)
                    # swap info (eg fields) of current op (eg, filter) and its predecessor (ie join)
                    new_node.data_metadata["output_fields"] = new_node.data_metadata["input_fields"] = last_node.data_metadata["input_left_fields"]
                    # rewire edges between current op and its predecessor and rewrite sub-lineage over pushdowned filter
                    new_node.set_left_parent(last_node.left_parent)
                    last_node.set_left_parent(cls.rewrite_plan(new_node))
                    pushdown_flag = True
                # push filter into the right sub-lineage
                if input_column in right_fields:
                    new_node = LineageNode(op_name="filter", op_metadata=node.op_metadata, data_metadata=node.data_metadata)
                    # swap info (eg fields) of current op, filter, and its predecessor (ie join)
                    new_node.data_metadata["output_fields"] = new_node.data_metadata["input_fields"] = last_node.data_metadata["input_right_fields"]
                    # rewire edges between current op and its predecessor and rewrite sub-lineage over pushdowned filter
                    new_node.set_left_parent(last_node.right_parent)
                    last_node.set_right_parent(cls.rewrite_plan(new_node))
                    pushdown_flag = True
                if pushdown_flag:
                    del node
                    return last_node
                else:
                    return node
            
            else:
                fields = last_node.data_metadata["input_fields"]
                if input_column in fields:
                    # swap info (eg fields) of current op, filter, and its predecessor (eg map)
                    node.data_metadata["output_fields"] = node.data_metadata["input_fields"] = last_node.data_metadata["input_fields"]
                    # rewire edges around current op and its predecessor, and rewrite sub-lineage over pushdowned filter
                    node.set_left_parent(last_node.left_parent)
                    last_node.set_left_parent(cls.rewrite_plan(node))
                    return last_node
                else:
                    return node
                
        elif node.op_name == "join":
            node.set_left_parent(cls.rewrite_plan(node.left_parent))
            node.set_right_parent(cls.rewrite_plan(node.right_parent))
            return node
        
        elif node.op_name == "map" or node.op_name == "reduce":
            node.set_left_parent(cls.rewrite_plan(node.left_parent))
            return node
        
        else:
            return node


class NonLLMPushdown:
    @classmethod
    def rewrite_plan(cls, node: LineageNode) -> LineageNode:
        if node.op_name in ["map", "filter", "reduce"]:
            last_node = node.left_parent
            last_node = cls.rewrite_plan(last_node)
            func = node.op_metadata.get("func", None)
            input_column = node.op_metadata["input_column"]

            if func and input_column not in last_node.data_metadata.get("output_fields", []):
                # push non-LLM ops down if they have a UDF and their action scope is not included in the output_fields of their ancestors
                new_node = LineageNode(op_name=node.op_name, op_metadata=node.op_metadata, data_metadata=node.data_metadata)
                # swap info (eg fields) of current op and its predecessor
                new_node.data_metadata["input_fields"] = last_node.data_metadata["input_fields"]
                new_node.data_metadata["output_fields"] = (
                    list(set(last_node.data_metadata["input_fields"] + [node.op_metadata["output_column"]]))
                    if "output_column" in node.op_metadata else 
                    last_node.data_metadata["input_fields"]
                )
                last_node.data_metadata["input_fields"] = new_node.data_metadata["output_fields"]
                last_node.data_metadata["output_fields"] = (
                    list(set(last_node.data_metadata["input_fields"] + [last_node.op_metadata["output_column"]]))
                    if "output_column" in last_node.op_metadata else 
                    last_node.data_metadata["input_fields"]
                )
                new_node.set_left_parent(last_node.left_parent)
                last_node.set_left_parent(cls.rewrite_plan(new_node))
                del node
                return last_node
            else:
                return node
        
        elif node.op_name == "join":
            node.set_left_parent(cls.rewrite_plan(node.left_parent))
            node.set_right_parent(cls.rewrite_plan(node.right_parent))
            return node
        
        else:
            return node
