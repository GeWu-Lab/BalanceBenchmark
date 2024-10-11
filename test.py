from itertools import combinations
def generate_all_combinations(input_list: list[str], include_empty: bool = True):
    """
    生成输入列表的所有可能组合。
    
    :param input_list: 输入列表
    :return: 包含所有可能组合的列表
    """
    all_combinations = []
    
    # 生成长度从 0 到 len(input_list) 的所有组合
    start_range = 1 if include_empty else 0
    for r in range(start_range, len(input_list) + 1):
        all_combinations.extend(combinations(input_list, r))
    
    # 将组合转换为列表
    return [list(combo) for combo in all_combinations]

print(generate_all_combinations([1,2,3]))