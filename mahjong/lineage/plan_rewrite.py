from mahjong.lineage.lineage import LineageNode, LineageOpNode, LineageDataNode


def get_last_op_node(node: LineageNode):
    if isinstance(node, LineageDataNode):
            return node.parent
    op_nodes = []
    for p in node.parent:
        last_nodes = get_last_op_node(p)
        op_nodes.extend(last_nodes)
    return op_nodes


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
        if len(op.parent) > 1:
            return None
        
        # rule 1: combine two filter ops into one
        if op.op_name == "filter":
            last_op = get_last_op_node(op)[0]
            if last_op.op_name == "filter":
                new_op = cls()._fuse_filter_ops(op, last_op)
                if new_op is None:
                    return
                new_op.child = op.child
                new_op.parent = last_op.parent
                for p in last_op.parent:
                    p.child.remove(last_op)
                    p.add_child(new_op)
                for c in op.child:
                    c.parent.remove(op)
                    c.add_child(new_op)
                del last_op
                del op
                return new_op


class FilterPushdown:
    
    @classmethod
    def rewrite_op(cls, op: LineageOpNode):
        if op.op_name != "filter":
            return None
        
        last_op: LineageOpNode = get_last_op_node(op)[0]
        last_data: LineageDataNode = last_op.child[0]
        curr_data: LineageDataNode = op.child[0]
        
        # if the target column of the filter op depends on the last op, do not push it down
        if op.input_column not in last_data.columns:
            return None
        
        # exchange the columns in the last data node and current data node
        curr_data.columns = last_data.columns

        # exchange the parent-child relationship between current node and last node
        last_data.child = curr_data.child
        curr_data.child = [last_op]
        op.parent = last_op.parent
        last_op.parent = [curr_data]

        # rewire the parent-child relationship for nodes before last node and nodes after curr node
        for p in last_op.parent:
            p.child.remove(last_op)
            p.add_child(op)
        for c in curr_data.child:
            c.parent.remove(curr_data)
            c.add_parent(last_data)

        return last_op
