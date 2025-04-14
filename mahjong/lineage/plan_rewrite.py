from mahjong.lineage.lineage import LineageOpNode


class OpFusion:

    def _fuse_filter_ops(self, child: LineageOpNode, parent: LineageOpNode):
        # if the consecutive ops are not applied on the same schema, refuse filter fusion
        if child.input_schema != parent.input_schema:
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
            input_schema=child.input_schema,
        )
        return new_op_node

    def _fuse_map_ops(self, child: LineageOpNode, parent: LineageOpNode):
        pass

    def rewrite_op(self, op: LineageOpNode):
        if len(op.parent) > 1:
            return
        
        # rule 1: combine two filter ops into one
        if op.op_name == "filter":
            last_op = op.parent[0]
            if last_op.op_name == "filter":
                new_op = self._fuse_filter_ops(op, last_op)
                if new_op is None:
                    return
                new_op.child = op.child
                new_op.parent = last_op.parent
                for p in last_op.parent:
                    p.child.remove(last_op)
                    p.child.append(new_op)
                for c in op.child:
                    c.parent.remove(op)
                    c.parent.append(new_op)
            pass


class FilterPushdown:
    
    def rewrite_op(self, op: LineageOpNode):
        if op.op_name != "filter":
            return
        
        
