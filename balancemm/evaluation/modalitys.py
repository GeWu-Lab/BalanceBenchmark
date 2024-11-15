from itertools import combinations
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import logging
from copy import deepcopy
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

def Calculate_Shapley(trainer, model, CalcuLoader: Dataset, logger: logging.Logger, include_empty: bool = False, conduct: bool = True) -> dict[str: float]:
    if conduct:  
        modalitys = model.modalitys
        Shapley = defaultdict(int) ##default is 0
        res_cahce = defaultdict(lambda:float('inf')) ## store the middle results
        for modality in modalitys:
            temp_modalitys = list(modalitys)
            # if include_empty:
            #     temp_modalitys.append([]) wrong
            temp_modalitys.remove(modality)
            combinations = generate_all_combinations(temp_modalitys, include_empty = True)
            if include_empty:
                combinations.append([])
            for combo in combinations:
                indentifer = tuple(sorted(combo))
                if res_cahce[indentifer] == float('inf'):
                    _, v_combo = trainer.val_loop(model = model, val_loader= CalcuLoader, limit_modalitys= combo.copy())
                    res_cahce[indentifer] = v_combo
                else:
                    v_combo = res_cahce[indentifer]
                if modality not in combo:
                    add_combo = combo.copy()
                    add_combo.append(modality)
                    add_combo = sorted(add_combo)
                    indentifer = tuple(add_combo)
                    if res_cahce[indentifer] == float('inf'):
                        _, v_add = trainer.val_loop(model = model, val_loader= CalcuLoader, limit_modalitys= add_combo)
                        res_cahce[indentifer] = v_add
                    else:
                        v_add = res_cahce[indentifer]
                else:
                    v_add = v_combo
                res = (v_add['acc']['output'] - v_combo['acc']['output'])
                print(f'{modality} acc: {res}')
                Shapley[modality] += (v_add['acc']['output'] - v_combo['acc']['output'])
        logger.info(Shapley)
        return Shapley
    else:
        return 