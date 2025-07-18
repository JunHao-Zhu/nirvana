from nirvana.lineage.abstractions import LineageNode, LineageOpNode, LineageDataNode


def rewrite_plan(plan: LineageNode):
    node_stack = [plan]
    while len(node_stack) > 0:
        node = node_stack.pop()
        if isinstance(node, LineageDataNode):
            for p in node.parent:
                node_stack.append(p)
            continue
        if node.is_optimized:
            continue
        if len(node.parent) == 0:
            node.is_optimized = True
            continue
        optimized_node = FilterPushdown.rewrite_op(node)
        if optimized_node is not None:
            # if the optimization is successful, continue to optimize the parent node
            if len(node_stack) < 1:
                node_stack.append(optimized_node)
            for p in optimized_node.parent:
                node_stack.append(p)
        else:
            node.is_optimized = True
            for p in node.parent:
                node_stack.append(p)
    
    def traverse(node: LineageNode):
        if len(node.child) > 0:
            last_node = traverse(node.child[0])
        else:
            last_node = node
        return last_node
    optimized_plan = traverse(plan)
    return optimized_plan


def get_last_op_node(node: LineageNode):
    if isinstance(node, LineageDataNode):
            return node.parent
    op_nodes = []
    for p in node.parent:
        last_nodes = get_last_op_node(p)
        op_nodes.extend(last_nodes)
    return op_nodes


class FilterPushdown:
    # TODO: enable filter pushdown on join ops
    @classmethod
    def rewrite_op(cls, op: LineageOpNode):
        if op.op_name != "filter" or op.is_optimized:
            return None
        
        last_op: LineageOpNode = get_last_op_node(op)[0]
        last_data: LineageDataNode = last_op.child[0]
        curr_data: LineageDataNode = op.child[0]
        
        # if the target column of the filter op depends on the last op, do not push it down
        if op.input_column not in last_data.columns:
            return None
        
        # exchange the columns in the last data node and current data node
        tmp = curr_data.columns
        curr_data.columns = last_data.columns
        last_data.columns = tmp

        # rewire the parent-child relationship for nodes before last node and nodes after curr node
        for p in last_op.parent:
            p.remove_child(last_op)
            p.add_child(op)
        for c in curr_data.child:
            c.remove_parent(curr_data)
            c.add_parent(last_data)

        # exchange the parent-child relationship between current node and last node
        last_data.set_child(curr_data.child)
        curr_data.set_child([last_op])
        op.set_parent(last_op.parent)
        last_op.set_parent([curr_data])

        return last_op


class OpFusion:

    def _fuse_filter_ops(self, child: LineageOpNode, parent: LineageOpNode):
        # if the consecutive ops are not applied on the same field, refuse filter fusion
        if child.input_column != parent.input_column:
            return None
        
        # create the new user instruction
        new_instruction = []
        if isinstance(parent.user_instruction, str):
            new_instruction.append(parent.user_instruction)
        elif isinstance(parent.user_instruction, list):
            new_instruction.extend(parent.user_instruction)
        if isinstance(child.user_instruction, str):
            new_instruction.append(child.user_instruction)
        elif isinstance(child.user_instruction, list):
            new_instruction.extend(child.user_instruction)

        new_op_node = LineageOpNode(
            op_name="filter",
            user_instruction=new_instruction,
            input_column=child.input_column,
        )
        return new_op_node

    def _fuse_map_ops(self, child: LineageOpNode, parent: LineageOpNode):
        pass
    
    @classmethod
    def rewrite_op(cls, op: LineageOpNode):
        # rule 1: combine two filter ops into one
        if op.op_name == "filter":
            last_op = get_last_op_node(op)[0]
            if last_op.op_name == "filter":
                new_op = cls()._fuse_filter_ops(op, last_op)
                if new_op is None:
                    return None
                new_op.set_child(op.child)
                new_op.set_parent(last_op.parent)
                for p in last_op.parent:
                    p.remove_child(last_op)
                    p.add_child(new_op)
                for c in op.child:
                    c.remove_parent(op)
                    c.add_parent(new_op)
                del last_op
                del op
                return new_op
            
        return None
