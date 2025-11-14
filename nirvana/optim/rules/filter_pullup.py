from nirvana.lineage.abstractions import LineageNode


class FilterPullup:
    # TODO: fix bugs when pulling up multiple filters in a branch
    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name == "join":
            # First, transform both branches below the join
            node.set_left_parent(cls.transform(node.left_parent))
            node.set_right_parent(cls.transform(node.right_parent))

            # Attempt to pull up a filter from the left branch if safe
            left_parent = node.left_parent
            pulled_filters = []
            if left_parent and left_parent.op_name == "filter":
                # it is always safe to pull up a filter
                new_filter = LineageNode(
                    op_name="filter",
                    op_kwargs=left_parent.operator.op_kwargs,
                    node_fields=left_parent.node_fields.model_dump(),
                )

                # Rewire: join connects to the filter's previous parent; filter connects after join
                node.set_left_parent(left_parent.left_parent)
                pulled_filters.append(new_filter)
                del left_parent

            # Attempt to pull up a filter from the right branch if safe
            right_parent = node.right_parent
            if right_parent and right_parent.op_name == "filter":
                # it is always safe to pull up a filter
                new_filter = LineageNode(
                    op_name="filter",
                    op_kwargs=right_parent.operator.op_kwargs,
                    node_fields=right_parent.node_fields.model_dump(),
                )

                # Rewire: join connects to the filter's previous parent; filter connects after join
                node.set_right_parent(right_parent.left_parent)
                del right_parent
                pulled_filters.append(new_filter)
            
            prev_node = node
            for filter in pulled_filters:
                # The input_left_fields and output_fields of join remains unchanged
                # the only things to change are input_fields and output_fields of filter
                filter.node_fields.left_input_fields = prev_node.node_fields.output_fields
                filter.node_fields.output_fields = filter.node_fields.left_input_fields
                filter.set_left_parent(prev_node)
                prev_node = filter
            return prev_node

        elif node.op_name in ["map", "filter"]:
            # First, transform the subtree below
            node.set_left_parent(cls.transform(node.left_parent))

            parent = node.left_parent

            if parent.op_name == "filter":
                # It's always safe to pull up a filter
                new_filter = LineageNode(
                    op_name="filter",
                    op_kwargs=parent.operator.op_kwargs,
                    node_fields=parent.node_fields.model_dump(),
                )

                # The input_fields and output_fields of the current node are not affected
                # the only things to change are input_fields and output_fields of filter
                new_filter.node_fields.left_input_fields = node.node_fields.output_fields
                new_filter.node_fields.output_fields = new_filter.node_fields.left_input_fields

                # Rewire: node connects to map's previous parent; map connects after node
                node.set_left_parent(parent.left_parent)
                new_filter.set_left_parent(node)
                del parent
                return new_filter

            return node

        elif node.op_name == "reduce":
            node.set_left_parent(cls.transform(node.left_parent))
            return node

        else:
            return node
