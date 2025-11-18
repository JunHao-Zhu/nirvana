from nirvana.lineage.abstractions import LineageNode


class NonLLMPushdown:
    @classmethod
    def check_pattern(cls, node: LineageNode, dependencies: list[str], generated_fields: list[str]) -> bool:
        return node.operator.has_udf() and all([field not in generated_fields for field in dependencies])
    
    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name in ["map", "filter"]:
            last_node = cls.transform(node.left_child)

            if last_node.op_name in ["scan", "join"]:
                node.set_left_child(last_node)
                return node
            
            dependencies = node.operator.dependencies
            generated_fields = last_node.operator.generated_fields
            if cls.check_pattern(node, dependencies, generated_fields):
                # push non-LLM ops down if they have a UDF and their action scope is not included in the output_fields of their ancestors
                new_node = LineageNode(op_name=node.op_name, op_kwargs=node.operator.op_kwargs, node_fields=node.node_fields.model_dump())
                # swap info (eg fields) of current op and its predecessor
                new_node.node_fields.left_input_fields = last_node.node_fields.left_input_fields
                new_node.node_fields.output_fields = list(
                    set(new_node.node_fields.left_input_fields + new_node.operator.generated_fields)
                )
                last_node.node_fields.left_input_fields = new_node.node_fields.output_fields
                last_node.node_fields.output_fields = list(
                    set(last_node.node_fields.left_input_fields + last_node.operator.generated_fields)
                )
                new_node.set_left_child(last_node.left_child)
                last_node.set_left_child(cls.transform(new_node))
                del node
                return last_node
            else:
                node.set_left_child(last_node)
                return node
        
        elif node.op_name == "join":
            node.set_left_child(cls.transform(node.left_child))
            node.set_right_child(cls.transform(node.right_child))
            return node

        elif node.op_name == "reduce":
            node.set_left_child(cls.transform(node.left_child))
            return node
        
        else:
            return node
