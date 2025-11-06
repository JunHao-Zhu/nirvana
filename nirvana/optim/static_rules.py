from nirvana.lineage.abstractions import LineageNode


class FilterPushdown:
    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name == "filter":
            last_node = node.left_parent
            last_node = cls.transform(last_node)
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
                    last_node.set_left_parent(cls.transform(new_node))
                    pushdown_flag = True
                # push filter into the right sub-lineage
                if input_column in right_fields:
                    new_node = LineageNode(op_name="filter", op_metadata=node.op_metadata, data_metadata=node.data_metadata)
                    # swap info (eg fields) of current op, filter, and its predecessor (ie join)
                    new_node.data_metadata["output_fields"] = new_node.data_metadata["input_fields"] = last_node.data_metadata["input_right_fields"]
                    # rewire edges between current op and its predecessor and rewrite sub-lineage over pushdowned filter
                    new_node.set_left_parent(last_node.right_parent)
                    last_node.set_right_parent(cls.transform(new_node))
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
                    last_node.set_left_parent(cls.transform(node))
                    return last_node
                else:
                    return node
                
        elif node.op_name == "join":
            node.set_left_parent(cls.transform(node.left_parent))
            node.set_right_parent(cls.transform(node.right_parent))
            return node
        
        elif node.op_name == "map" or node.op_name == "reduce":
            node.set_left_parent(cls.transform(node.left_parent))
            return node
        
        else:
            return node


class NonLLMPushdown:
    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name in ["map", "filter"]:
            last_node = node.left_parent
            last_node = cls.transform(last_node)
            func = node.op_metadata.get("func", None)
            input_column = node.op_metadata["input_column"]

            if func and input_column not in last_node.node_input_metadata.generations:
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
                last_node.set_left_parent(cls.transform(new_node))
                del node
                return last_node
            else:
                return node
        
        elif node.op_name == "join":
            node.set_left_parent(cls.transform(node.left_parent))
            node.set_right_parent(cls.transform(node.right_parent))
            return node

        elif node.op_name == "reduce":
            node.set_left_parent(cls.transform(node.left_parent))
            return node
        
        else:
            return node


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
                    op_metadata=left_parent.op_metadata,
                    data_metadata=left_parent.data_metadata,
                )

                # The input_left_fields and output_fields of join remains unchanged
                # the only things to change are input_fields and output_fields of filter
                new_filter.data_metadata["input_fields"] = node.data_metadata["output_fields"]
                new_filter.data_metadata["output_fields"] = new_filter.data_metadata["input_fields"]

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
                    op_metadata=right_parent.op_metadata,
                    data_metadata=right_parent.data_metadata,
                )

                # The input_right_fields and output_fields of join remains unchanged
                # the only things to change are input_fields and output_fields of filter
                new_filter.data_metadata["input_fields"] = node.data_metadata["output_fields"]
                new_filter.data_metadata["output_fields"] = new_filter.data_metadata["input_fields"]

                # Rewire: join connects to the filter's previous parent; filter connects after join
                node.set_right_parent(right_parent.left_parent)
                del right_parent
                pulled_filters.append(new_filter)
            
            prev_node = node
            for filter in pulled_filters:
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
                    op_metadata=parent.op_metadata,
                    data_metadata=parent.data_metadata,
                )

                # The input_fields and output_fields of the current node are not affected
                # the only things to change are input_fields and output_fields of filter
                new_filter.data_metadata["input_fields"] = node.data_metadata["output_fields"]
                new_filter.data_metadata["output_fields"] = new_filter.data_metadata["input_fields"]

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


class MapPullup:
    # TODO: fix bugs when pulling up multiple maps in a branch
    @classmethod
    def transform(cls, node: LineageNode) -> LineageNode:
        if node.op_name == "join":
            # First, transform both branches below the join
            node.set_left_parent(cls.transform(node.left_parent))
            node.set_right_parent(cls.transform(node.right_parent))

            # Attempt to pull up a map from the left branch if safe
            left_parent = node.left_parent
            pulled_maps = []
            if left_parent and left_parent.op_name == "map":
                map_output_col = left_parent.op_metadata.get("output_column", None)
                if map_output_col is not None:
                    # If the join does not reference the produced column from the left side, it is safe to pull up
                    input_left_fields = node.data_metadata.get("input_left_fields", [])
                    if map_output_col not in input_left_fields:
                        new_map = LineageNode(
                            op_name="map",
                            op_metadata=left_parent.op_metadata,
                            data_metadata=left_parent.data_metadata,
                        )

                        # Update join's left input fields to what the map previously consumed
                        node.data_metadata["input_left_fields"] = left_parent.data_metadata["input_fields"]

                        # Establish the join output fields (union of both sides)
                        join_output_fields = list(
                            set(node.data_metadata.get("input_left_fields", []) + node.data_metadata.get("input_right_fields", []))
                        )
                        node.data_metadata["output_fields"] = join_output_fields

                        # The pulled-up map now consumes the join's outputs and adds its produced column
                        new_map.data_metadata["input_fields"] = node.data_metadata.get("output_fields", join_output_fields)
                        new_map.data_metadata["output_fields"] = list(
                            set(new_map.data_metadata["input_fields"] + [map_output_col])
                        )

                        # Rewire: join connects to the map's previous parent; map connects after join
                        node.set_left_parent(left_parent.left_parent)
                        del left_parent
                        pulled_maps.append(new_map)

            # Attempt to pull up a map from the right branch if safe
            right_parent = node.right_parent
            if right_parent and right_parent.op_name == "map":
                map_output_col = right_parent.op_metadata.get("output_column", None)
                if map_output_col is not None:
                    input_right_fields = node.data_metadata.get("input_right_fields", [])
                    if map_output_col not in input_right_fields:
                        new_map = LineageNode(
                            op_name="map",
                            op_metadata=right_parent.op_metadata,
                            data_metadata=right_parent.data_metadata,
                        )

                        # Update join's right input fields to what the map previously consumed
                        node.data_metadata["input_right_fields"] = right_parent.data_metadata["input_fields"]

                        # Establish the join output fields (union of both sides)
                        join_output_fields = list(
                            set(node.data_metadata.get("input_left_fields", []) + node.data_metadata.get("input_right_fields", []))
                        )
                        node.data_metadata["output_fields"] = join_output_fields

                        # The pulled-up map now consumes the join's outputs and adds its produced column
                        new_map.data_metadata["input_fields"] = node.data_metadata.get("output_fields", join_output_fields)
                        new_map.data_metadata["output_fields"] = list(
                            set(new_map.data_metadata["input_fields"] + [map_output_col])
                        )

                        # Rewire: join connects to the map's previous parent; map connects after join
                        node.set_right_parent(right_parent.left_parent)
                        del right_parent
                        pulled_maps.append(new_map)
            prev_node = node
            for map in pulled_maps:
                map.set_left_parent(prev_node)
                prev_node = map
            return prev_node

        elif node.op_name in ["map", "filter"]:
            # First, transform the subtree below
            node.set_left_parent(cls.transform(node.left_parent))

            parent = node.left_parent

            # If the parent is a map and the current node does not reference the map's produced column,
            # pull the map up (i.e., after the current node)
            if parent.op_name == "map":
                map_output_col = parent.op_metadata.get("output_column", None)
                if map_output_col is None:
                    return node

                child_input_col = node.op_metadata.get("input_column", None)
                # Only safe to pull up when the child doesn't reference the produced column
                if child_input_col is not None and child_input_col != map_output_col:
                    new_map = LineageNode(
                        op_name="map",
                        op_metadata=parent.op_metadata,
                        data_metadata=parent.data_metadata,
                    )

                    # Update metadata to reflect the new ordering
                    # Child now receives what the map used to receive
                    node.data_metadata["input_fields"] = parent.data_metadata["input_fields"]
                    node.data_metadata["output_fields"] = (
                        list(set(node.data_metadata["input_fields"] + [node.op_metadata["output_column"]]))
                        if "output_column" in node.op_metadata else node.data_metadata["input_fields"]
                    )

                    # The pulled-up map now consumes the child's outputs
                    new_map.data_metadata["input_fields"] = node.data_metadata["output_fields"]
                    new_map.data_metadata["output_fields"] = list(
                        set(new_map.data_metadata["input_fields"] + [map_output_col])
                    )

                    # Rewire: node connects to map's previous parent; map connects after node
                    node.set_left_parent(parent.left_parent)
                    new_map.set_left_parent(node)
                    del parent
                    return new_map

            return node

        elif node.op_name == "reduce":
            node.set_left_parent(cls.transform(node.left_parent))
            return node

        else:
            return node
