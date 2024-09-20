from ..trainer.base_trainer import BaseTrainer
from ..models.avclassify_model import BaseClassifierModel
from itertools import combinations
from collections import defaultdict
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

def Calculate_sharply(trainer: BaseTrainer, model: BaseClassifierModel, CalcuLoader) -> dict[str: float]:
    modalitys = model.modalitys
    sharply = defaultdict(int) ##default is 0
    for modality in modalitys:
        temp_modalitys = modalitys.copy()
        temp_modalitys.remove(modality)
        combinations = generate_all_combinations(temp_modalitys, include_empty = True)
        for combo in combinations:
            v_combo = trainer.val_loop(model = model, val_loader= CalcuLoader, limit_modalitys= combo)
            if modality not in combo:
                add = combo.append(modality)
                v_add = trainer.val_loop(model = model, val_loader= CalcuLoader, limit_modalitys= add)
            else:
                v_add = v_combo
            sharply[modality] += (v_add['output'] - v_combo['output'])
    return sharply