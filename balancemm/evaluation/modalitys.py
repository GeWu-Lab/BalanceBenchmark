from itertools import combinations
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import logging
from copy import deepcopy
from tqdm import tqdm
from math import factorial
import torch
def generate_all_combinations(input_list: list[str], include_empty: bool = True):
    """
    生成输入列表的所有可能组合。
    
    :param input_list: 输入列表
    :return: 包含所有可能组合的列表
    """
    all_combinations = []
    
    # 生成长度从 0 到 len(input_list) 的所有组合
    start_range = 0 if include_empty else 1
    for r in range(start_range, len(input_list) + 1):
        all_combinations.extend(combinations(input_list, r))
    # 将组合转换为列表
    return [list(combo) for combo in all_combinations]

def Calculate_Shapley(trainer, model, CalcuLoader: Dataset, logger: logging.Logger, conduct: bool = True) -> dict[str: float]:
    
    if conduct:  
        modalitys = model.modalitys
        n = len(modalitys)
        Shapley = defaultdict(float) ##default is 0
        res_cahce = defaultdict(lambda:float('inf')) ## store the middle results
        for modality in modalitys:
            temp_modalitys = list(modalitys)
            # if include_empty:
            #     temp_modalitys.append([]) wrong
            temp_modalitys.remove(modality)
            combinations = generate_all_combinations(temp_modalitys, include_empty = True)
            for combo in combinations:
                S_size = len(combo)
                indentifer = tuple(sorted(combo))
                if res_cahce[indentifer] == float('inf'):
                    with torch.no_grad():
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
                        with torch.no_grad():
                            _, v_add = trainer.val_loop(model = model, val_loader= CalcuLoader, limit_modalitys= add_combo)
                        res_cahce[indentifer] = v_add
                    else:
                        v_add = res_cahce[indentifer]
                else:
                    v_add = v_combo
                Shapley[modality] += (factorial(S_size) * factorial(n - S_size - 1)) / factorial(n)*(v_add['acc']['output'] - v_combo['acc']['output'])
        logger.info(Shapley)
        return Shapley
    else:
        return 
    
def Calculate_Shapley_Sample(trainer, model, CalcuLoader: Dataset, logger: logging.Logger,conduct: bool = True,is_print: bool = False) -> dict[str: float]:
    if not conduct:
        return None
    modalitys = model.modalitys  
    n = len(modalitys)  
    Shapley = {modality: {} for modality in modalitys}  
    res_cache = defaultdict(lambda: None)  
    
    for batch_idx, batch in tqdm(enumerate(CalcuLoader)):
        label = batch['label'].to(model.device)
        batch_size = len(label)

        all_combinations = generate_all_combinations(modalitys, include_empty=True)
        for combo in all_combinations:
            identifier = tuple(sorted(combo))
            if res_cache[identifier] is None:
                if not combo:
                    res_cache[identifier] = torch.zeros(batch_size, dtype=torch.bool)
                else:
                    with torch.no_grad():
                        model.validation_step(batch, batch_idx,limit_modality=combo)
                    res_cache[identifier] = (model.pridiction['output'] == label)
        
        for i in range(batch_size):
            sample_idx = int(batch['idx'][i])
 
            for modality in modalitys:
                shapley_value = 0.0
                temp_modalitys = [m for m in modalitys if m != modality]
                combinations = generate_all_combinations(temp_modalitys, include_empty=True)
                
                for combo in combinations:
                    S_size = len(combo)
                    v_combo = res_cache[tuple(sorted(combo))][i]

                    add_combo = sorted(combo + [modality])
                    v_add = res_cache[tuple(add_combo)][i]

                    weight = (factorial(S_size) * factorial(n - S_size - 1)) / factorial(n)
                    marginal_contribution = float(v_add) - float(v_combo)
                    shapley_value += weight * marginal_contribution
                
                Shapley[modality][sample_idx] = shapley_value
                

    return Shapley