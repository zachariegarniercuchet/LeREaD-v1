

def protected_levenshtein_distance(original, derived, is_protected_func):
    """
    Protected tokens (determined by is_protected_func) can NEVER be modified.
    They can only be deleted. This prevents the algorithm from trying to modify
    protected tokens to match the original.
    
    Args:
        original: The target list (what we want to achieve)
        derived: The source list (what we start with)
        is_protected_func: Function that takes a token and returns True if it should be protected
    
    Returns:
        tuple: (distance, operations)
            - distance: minimum number of operations needed
            - operations: list of operations to apply to derived to get original
                         Each operation is a tuple: ('insert', index, value), ('delete', index), or ('modify', index, value)
    
    Example:
        original = ['Act', ',', '\\n', 'S']
        derived = ['Act', '<sep>', ',', ' ', '<sep>', 'S']
        is_protected = lambda token: token == '<sep>'
        -> [('delete', 1), ('modify', 2, '\\n'), ('delete', 4)]
    """
    n = len(original)
    m = len(derived)
    
    # Create DP table: dp[i][j] = min operations to transform derived[:j] into original[:i]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Base cases
    for i in range(n + 1):
        dp[i][0] = i  # Need i insertions to get from empty list to original[:i]
    for j in range(m + 1):
        dp[0][j] = j  # Need j deletions to get from derived[:j] to empty list
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if original[i-1] == derived[j-1]:
                # No operation needed
                dp[i][j] = dp[i-1][j-1]
            else:
                # Calculate costs for each operation
                insert_cost = dp[i-1][j] + 1  # Insert original[i-1] at position j in derived
                delete_cost = dp[i][j-1] + 1  # Delete derived[j-1]
                
                # Modify is only allowed if derived[j-1] is NOT protected
                if is_protected_func(derived[j-1]):
                    # Cannot modify protected tokens - set modify cost to infinity
                    modify_cost = float('inf')
                else:
                    modify_cost = dp[i-1][j-1] + 1  # Modify derived[j-1] to original[i-1]
                
                dp[i][j] = min(insert_cost, delete_cost, modify_cost)
    
    # Backtrack to find operations
    operations = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i == 0:
            # Need to delete all remaining items from derived
            operations.append(('delete', j-1))
            j -= 1
        elif j == 0:
            # Need to insert all remaining items from original
            operations.append(('insert', 0, original[i-1]))
            i -= 1
        elif original[i-1] == derived[j-1]:
            # No operation needed, items match
            i -= 1
            j -= 1
        else:
            # Find which operation was taken
            insert_cost = dp[i-1][j]
            delete_cost = dp[i][j-1]
            
            # Check modify cost (will be inf if derived[j-1] is protected)
            if is_protected_func(derived[j-1]):
                modify_cost = float('inf')
            else:
                modify_cost = dp[i-1][j-1]
            
            min_cost = min(insert_cost, delete_cost, modify_cost)
            
            if min_cost == modify_cost:
                # Modify operation
                operations.append(('modify', j-1, original[i-1]))
                i -= 1
                j -= 1
            elif min_cost == delete_cost:
                # Delete operation
                operations.append(('delete', j-1))
                j -= 1
            else:
                # Insert operation
                operations.append(('insert', j, original[i-1]))
                i -= 1
    
    # Reverse operations since we backtracked
    operations.reverse()
    
    # Adjust indices: operations are relative to the derived list as it's being transformed
    # We need to adjust indices to account for previous operations
    adjusted_operations = []
    offset = 0
    
    for op in operations:
        if op[0] == 'insert':
            adjusted_operations.append(('insert', op[1] + offset, op[2]))
            offset += 1
        elif op[0] == 'delete':
            adjusted_operations.append(('delete', op[1] + offset))
            offset -= 1
        else:  # modify
            adjusted_operations.append(('modify', op[1] + offset, op[2]))
    
    distance = dp[n][m]
    return distance, adjusted_operations


def apply_operations_safe(processed_tokens, operations, is_protected_func):
    """
    
    Args:
        processed_tokens: List of tokens to modify
        operations: list of operations
        is_protected_func: Function that takes a token and returns True if it should be protected
    
    Returns:
        result_tokens: Modified token list with operations applied
    
    Rules:
        - Insert: Always insert at the given index
        - Delete: Delete at index ONLY if it's not a protected token
                 If skipped, adjust all following operation indices by +1
        - Modify: Modify at index ONLY if it's not a protected token
    
    Key insight: When we skip a delete on a protected token, the list doesn't 
    change but the original operation assumed it would. So we need to shift subsequent 
    indices to compensate.
    """
    result_tokens = processed_tokens[:]
    
    # Adjust operations as we go
    adjusted_ops = list(operations)
    
    i = 0
    while i < len(adjusted_ops):
        op = adjusted_ops[i]
        op_type = op[0]
        
        if op_type == 'insert':
            # Always insert
            index = op[1]
            value = op[2]
            result_tokens.insert(index, value)
        
        elif op_type == 'delete':
            # Delete only if not protected
            index = op[1]
            if 0 <= index < len(result_tokens):
                if not is_protected_func(result_tokens[index]):
                    result_tokens.pop(index)
                else:
                    # Skipped deletion - adjust all following indices by +1
                    for j in range(i + 1, len(adjusted_ops)):
                        next_op = adjusted_ops[j]
                        if next_op[1] >= index:
                            if next_op[0] == 'insert':
                                adjusted_ops[j] = ('insert', next_op[1] + 1, next_op[2])
                            elif next_op[0] == 'delete':
                                adjusted_ops[j] = ('delete', next_op[1] + 1)
                            elif next_op[0] == 'modify':
                                adjusted_ops[j] = ('modify', next_op[1] + 1, next_op[2])
        
        elif op_type == 'modify':
            # Modify only if not protected
            index = op[1]
            value = op[2]
            if 0 <= index < len(result_tokens):
                if not is_protected_func(result_tokens[index]):
                    result_tokens[index] = value
        
        i += 1
    
    return result_tokens
