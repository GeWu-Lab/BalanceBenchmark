#!/bin/bash
lr=$1
device=$2
dataset=$3
# methods=("OGMTrainer" "AGMTrainer" "AMCoTrainer" "CMLTrainer" "GBlendingTrainer" "PMRTrainer" "MBSDTrainer" "MMCosineTrainer" "UMTTrainer" "GreedyTrainer")

alphas=("AGMTrainer" "OGMTrainer" "UMTTrainer" "GreedyTrainer") #greedy #UMT
# alphas=("OGMTrainer")
lams=("CMLTrainer")
scalings=("MMCosineTrainer")
unique=("PMRTrainer")
normals=("AMCoTrainer" "GBlendingTrainer" "MBSDTrainer" "baselineTrainer")
groups=(alphas lams unique scalings normals)
# methods=("MMCosineTrainer" "GreedyTrainer")
# methods=("AGMTrainer" "OGMTrainer" "GreedyTrainer")
methods=("OGMTrainer")
# methods=("AGMTrainer" "GBlendingTrainer" "OGMTrainer")
# 定义参数范围

alpha_scaler=(0.1 0.5 1.0 1.5)
target_accuracy=0.81  # 预设目标准确率

# # 日志文件
# log_file="training_log_$normals.txt"
# results_file="results.txt"

# # 创建或清空日志文件
# > "$log_file"
# > "$results_file"

# 记录日志的函数
# log() {
#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
# }

setup() {
    alpha_train=$1
    lr_train=$2
    lam_train=$3
    scaling_train=$4
    mu_train=$5
    eta_train=$6
    momentum_train=$7
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
    log "python -m balancemm --model $model_name --trainer $method --lr $lr_train --alpha $alpha_train --mu $mu_train --scaling $scaling_train --lam $lam_train --eta $eta_train --momentum $momentum_train --dataset $dataset --device $device"
    local accuracy=$(python -m balancemm --model $model_name --trainer $method --lr $lr_train --alpha $alpha_train --mu $mu_train --scaling $scaling_train --lam $lam_train --eta $eta_train --momentum $momentum_train --dataset $dataset --device $device| tail -n 1| grep -oP "best test acc is : \K[0-9.]+")
    echo "$accuracy" 
}

for method in "${methods[@]}"; do
    best_accuracy=0
    best_alpha=0
    best_lr=0
    best_scaling=0
    best_lam=0

    # 日志文件
    log_file="training_log_$method.txt"
    results_file="results_$method.txt"

    # 创建或清空日志文件
    >> "$log_file"
    >> "$results_file"

    # 记录日志的函数
    log() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1 " | tee -a "$log_file"
    }
    # 网格搜索参数
    log "===== 开始测试方法: $method ====="
    for group in "${groups[@]}"; do
        eval "group_methods=(\"\${$group[@]}\")"
        for group_method in "${group_methods[@]}"; do
            if [ "$group_method" == "$method" ]; then
                now_group=$group
                if [ "$method" == "AGMTrainer" ]; then
                    alpha_scaler=(0.1 0.5 1.0 1.5)
                fi
                if [ "$method" == "UMT" ]; then
                    alpha_scaler=(0.1 1 10 50)
                fi
                if [ "$method" == "OGMTrainer" ]; then
                    alpha_scaler=(0.05 0.3 0.5 1.0)
                fi
                if [ "$method" == "CMLTrainer" ]; then
                    # lam_scaler=(10 15 30 45)
                    lam_scaler=(0.2 0.3 0.5 1)
                fi
                if [ "$method" == "MMCosineTrainer" ]; then
                    # scaling_scaler=(5 10 20 40)
                    scaling_scaler=(0.2 0.3 0.5 1)
                fi
                if [ "$method" == "GreedyTrainer" ]; then
                    alpha_scaler=(0.001 0.005 0.01)
                fi 
                if [ "$method" == "PMRTrainer" ]; then
                    eta_scaler=(0.01 0.03 0.05)
                    mu_scaler=(0.01 0.05 0.1)
                    momentum_scaler=(0.3 0.5 0.8)
                fi
            fi
        done   
    done 
    if [ "$now_group" == "alphas" ]; then
        for alpha_now in "${alpha_scaler[@]}"; do
            # 运行训练并获取结果
            lr_now=$lr
            setup "$alpha_now" "$lr_now" "0" "0" "0" "0" "0"
            accuracy=$(run_training "$method")
            # 记录结果
            echo "$method,$alpha_now,$lr_now,$accuracy" >> "$results_file"
            
            # 更新最佳结果
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_alpha=$alpha_now
                best_lr=$lr
            fi
            
            log "Alpha: $alpha_now, LR: $lr, Accuracy: $accuracy"
            
            # 如果达到目标准确率，提前结束当前方法的搜索
            if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                log "达到目标准确率，提前结束搜索"
                log "方法 $method 的最佳结果:"
                log "Best Alpha: $best_alpha"
                log "Best Learning Rate: $best_lr"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
                log "========================"
                break 1  # 跳出两层循环
            fi
        done
    elif [ "$now_group" == "normals" ]; then
        for j in {0}; do
            # 运行训练并获取结果
            lr_now=$lr
            setup "0" "$lr_now" "0" "0" "0" "0" "0"
            accuracy=$(run_training "$method")
            # 记录结果
            echo "$method,"0",$lr_now,$accuracy" >> "$results_file"
            
            # 更新最佳结果
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_alpha="0"
                best_lr=$lr_now
            fi
            
            log "LR: $lr_now, Accuracy: $accuracy"
            
            # 如果达到目标准确率，提前结束当前方法的搜索
            if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                log "达到目标准确率，提前结束搜索"
                log "方法 $method 的最佳结果:"
                log "Best Learning Rate: $best_lr"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
                log "========================"
                break 1  # 跳出两层循环
            fi
        done
    elif [ "$now_group" == "lams" ]; then
        for lam_now in "${lam_scaler[@]}"; do
            # 运行训练并获取结果
            lr_now=$lr
            setup "0" "$lr_now" "$lam_now" "0" "0" "0" "0"
            accuracy=$(run_training "$method")
            # 记录结果
            echo "$method,"0",$lr_now,$accuracy" >> "$results_file"
            
            # 更新最佳结果
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_lam=$lam_now
                best_lr=$lr_now
            fi
            
            log "lam: $lam_now, LR: $lr_now, Accuracy: $accuracy"
            
            # 如果达到目标准确率，提前结束当前方法的搜索
            if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                log "达到目标准确率，提前结束搜索"
                log "方法 $method 的最佳结果:"
                log "Best lam: $best_lam"
                log "Best Learning Rate: $best_lr"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
                log "========================"
                break 1  # 跳出两层循环
            fi
        done
    elif [ "$now_group" == "scalings" ]; then
        for scaling_now in "${scaling_scaler[@]}"; do
            # 运行训练并获取结果
            lr_now=$lr
            setup "0" "$lr_now" "0" "$scaling_now" "0" "0" "0"
            accuracy=$(run_training "$method")
            # 记录结果
            echo "$method,"0",$lr_now,$accuracy" >> "$results_file"
            
            # 更新最佳结果
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_scaling=$scaling_now
                best_lr=$lr_now
            fi
            
            log "scaling: $scaling_now, LR: $lr_now, Accuracy: $accuracy"
            
            # 如果达到目标准确率，提前结束当前方法的搜索
            if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                log "达到目标准确率，提前结束搜索"
                log "方法 $method 的最佳结果:"
                log "Best scaling: $best_scaling"
                log "Best Learning Rate: $best_lr"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
                log "========================"
                break 1  # 跳出两层循环
            fi
        done
    elif [ "$now_group" == "unique" ]; then
        for i in "${!mu_scaler[@]}"; do
            # 运行训练并获取结果
            lr_now=$lr
            setup "0" "$lr_now" "0" "0" "${mu_scaler[i]}" "${eta_scaler[i]}" "${momentum_scaler[i]}"
            accuracy=$(run_training "$method")
            # 记录结果
            echo "$method,"0",$lr_now,$accuracy" >> "$results_file"
            
            # 更新最佳结果
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_mu=${mu_scaler[i]}
                best_eta=${eta_scaler[i]}
                best_momentum=${momentum_scaler[i]}
                best_lr=$lr_now
            fi
            
            log "scaling: $scaling_now, LR: $lr_now, Accuracy: $accuracy"
            
            # 如果达到目标准确率，提前结束当前方法的搜索
            if (( $(echo "$accuracy >= $target_accuracy" | bc -l) )); then
                log "达到目标准确率，提前结束搜索"
                log "方法 $method 的最佳结果:"
                log "Best mu: $best_mu"
                log "Best eta: $best_eta"
                log "Best momentum: $best_momentum"
                log "Best Learning Rate: $best_lr"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
                log "========================"
                break 1  # 跳出两层循环
            fi
        done
    fi
    # # 记录最佳结果
    log "方法 $method 的最佳结果:"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Best Accuracy: $best_accuracy" | tee -a "$log_file"
    log "========================"
    # 添加一些间隔时间，避免过于频繁的训练
    sleep 5
done