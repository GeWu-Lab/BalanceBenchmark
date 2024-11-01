#!/bin/bash
lr=$1
delta=$2
device=$3
alpha=0.2

# methods=("OGMTrainer" "AGMTrainer" "AMCoTrainer" "CMLTrainer" "GBlendingTrainer" "PMRTrainer" "MBSDTrainer" "MMCosineTrainer" "UMTTrainer" "GreedyTrainer")

alphas=("AGMTrainer") #greedy #UMT
# alphas=("OGMTrainer")
lams=("CMLTrainer")
scalings=("MMCosineTrainer")
unique=("PMRTrainer")
normals=("AMCoTrainer" "GBlendingTrainer" "MBSDTrainer")
groups=(normals)
# 定义参数范围

learning_rates=(0.001 0.01 0.1)
target_accuracy=0.9  # 预设目标准确率

# 日志文件
log_file="training_log_$normals.txt"
results_file="results.txt"

# 创建或清空日志文件
> "$log_file"
> "$results_file"

# 记录日志的函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

setup() {
    alpha_train=$1
    lr_train=$2
    lam_train=$3
    scaling_train=$4
    mu_train=$5
    eta_train=$6
}
# 运行训练并获取结果的函数
run_training() {
    local method=$1
    local model_name="BaseClassifier"
    log "开始训练 Method: $method, Alpha: $alpha_train, Learning Rate: $lr_train"
    # 读取结果（假设结果被写入到temp_results.txt）
    # 这里需要根据实际输出格式修改
    if [ $method == "AMCoTrainer" ];then
    model_name="BaseClassifier_AMCo"
    elif [ $method == "GreedyTrainer" ];then
    model_name="BaseClassifier_Greedy"
    fi
    echo "--model $model_name --trainer $method --lr $lr_train --alpha $alpha_train --mu $mu_train --scaling $scaling_train --lam $lam_train --eta $eta_train --dataset Mosei --device 0"
    local accuracy=$(python -m balancemm --model $model_name --trainer $method --lr $lr_train --alpha $alpha_train --mu $mu_train --scaling $scaling_train --lam $lam_train --eta $eta_train --dataset Mosei --device $device| tail -n 1| grep -oP "best val acc is : \K[0-9.]+")
    echo "$accuracy" 
}
for group in "${groups[@]}"; do
    eval "methods=(\"\${$group[@]}\")"
    for method1 in "${methods[@]}"; do
        log "===== 开始测试方法: $method1 ====="
        best_accuracy=0
        best_alpha=0
        best_lr=0
        # 网格搜索参数
        if [ "$group" == "alphas" ]; then
            gap_alpha=0.1
            for i in {0..5}; do
                for j in {0..5}; do
                    # 运行训练并获取结果
                    alpha_now=$(echo "$alpha + $i * $gap_alpha" | bc)
                    alpha_now=$(printf "%.3f" "$alpha_now")
                    lr_now=$(echo "$lr + $j * $delta" | bc)
                    lr_now=$(printf "%.3f" "$lr_now")
                    setup "$alpha_now" "$lr_now" "0" "0" "0" "0"
                    accuracy=$(run_training "$method1")
                    # 记录结果
                    echo "$method1,$alpha_now,$lr_now,$accuracy" >> "$results_file"
                    
                    # 更新最佳结果
                    if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                        best_accuracy=$accuracy
                        best_alpha=$alpha
                        best_lr=$lr
                    fi
                    
                    log "Alpha: $alpha, LR: $lr, Accuracy: $accuracy"
                    
                    # 如果达到目标准确率，提前结束当前方法的搜索
                    if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                        log "达到目标准确率，提前结束搜索"
                        break 2  # 跳出两层循环
                    fi
                done
            done
        elif [ "$group" == "normals" ]; then
            for j in {1..5}; do
                # 运行训练并获取结果
                lr_now=$(echo "$lr + $j * $delta" | bc)
                lr_now=$(printf "%.3f" "$lr_now")
                setup "0" "$lr_now" "0" "0" "0" "0"
                accuracy=$(run_training "$method1")
                # 记录结果
                echo "$method1,"0",$lr_now,$accuracy" >> "$results_file"
                
                # 更新最佳结果
                if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                    best_accuracy=$accuracy
                    best_alpha="0"
                    best_lr=$lr_now
                fi
                
                log "Alpha: $alpha, LR: $lr_now, Accuracy: $accuracy"
                
                # 如果达到目标准确率，提前结束当前方法的搜索
                if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                    log "达到目标准确率，提前结束搜索"
                    break 1  # 跳出两层循环
                fi
            done
        fi
    done
    # 记录最佳结果
    log "方法 $method1 的最佳结果:"
    log "Best Alpha: $best_alpha"
    log "Best Learning Rate: $best_lr"
    log "Best Accuracy: $best_accuracy"
    log "========================"
    
    # 添加一些间隔时间，避免过于频繁的训练
    sleep 5
done