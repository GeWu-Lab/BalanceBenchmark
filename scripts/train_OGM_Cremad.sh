#!/bin/bash
# lr = 0.01
python -m balancemm --Main_config.Trainer OGMTrainer --Main_config.device 2  --Main_config.dataset Cremad --Train.dataset Cremad --Test.dataset Cremad --train.parameter.base_lr 0.01
python -m balancemm --model BaseClassifier --trainer OGMTrainer --lr 0.003 --alpha 0.2 --dataset Mosei --device 0
python -m balancemm --model BaseClassifier --trainer GBlendingTrainer --lr 0.003 --alpha 1.0 --dataset Mosei --device 0
python -m balancemm --model BaseClassifier_AMCo --trainer AMCoTrainer --lr 0.003 --alpha 1.0 --dataset Mosei --device 1
python -m balancemm --model BaseClassifier --trainer unimodalTrainer --lr 0.003 --alpha 1.0 --dataset KS --device 0
python -m balancemm --model BaseClassifier --trainer unimodalTrainer --lr 0.003 --alpha 1.0 --dataset UMPC_Food --device 1
python -m balancemm --model BaseClassifier --trainer UMTTrainer --lr 0.003 --alpha 1.0 --dataset KS --device 0
python -m balancemm --model BaseClassifier --trainer UMTTrainer --lr 0.003 --alpha 1.0 --dataset UMPC_Food --device 0