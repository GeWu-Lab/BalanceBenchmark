import re

def parse_log_file(file_path):
    max_valid_acc = 0
    max_valid_acc_epoch = 0
    max_acc_a = 0
    max_acc_v = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            if "epoch" in line:
                epoch_match = re.search(r'epoch: (\d+)', line)
            if "valid_acc:" in line:
                # Extract all relevant metrics
                valid_acc_match = re.search(r'valid_acc: (\d+\.?\d*)', line)
                acc_a_match = re.search(r'acc_a: (\d+\.?\d*)', line)
                acc_v_match = re.search(r'acc_v: (\d+\.?\d*)', line)
                print(valid_acc_match)
                print(epoch_match)
                print(acc_a_match)
                print(acc_v_match)
                if valid_acc_match and epoch_match and acc_a_match and acc_v_match:
                    valid_acc = float(valid_acc_match.group(1))
                    epoch = int(epoch_match.group(1))
                    acc_a = float(acc_a_match.group(1))
                    acc_v = float(acc_v_match.group(1))
                    
                    if valid_acc > max_valid_acc:
                        max_valid_acc = valid_acc
                        max_valid_acc_epoch = epoch
                        max_acc_a = acc_a
                        max_acc_v = acc_v

    return max_valid_acc, max_valid_acc_epoch, max_acc_a, max_acc_v

# 使用示例
log_file_path = '/data/users/shaoxuan_xu/BalanceMM-OGM_4.11/experiments/OGM-Balance/train_20240517-132336/training.log'
max_acc, max_acc_epoch, corresponding_acc_a, corresponding_acc_v = parse_log_file(log_file_path)
print(f"Maximum valid_acc: {max_acc}")
print(f"Epoch with maximum valid_acc: {max_acc_epoch}")
print(f"Corresponding acc_a: {corresponding_acc_a}")
print(f"Corresponding acc_v: {corresponding_acc_v}")