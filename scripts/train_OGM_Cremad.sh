#!/bin/bash
# lr = 0.01
python -m balancemm --Main_config.Trainer OGMTrainer --Main_config.device 2  --Main_config.dataset Cremad --Train.dataset Cremad --Test.dataset Cremad --train.parameter.base_lr 0.01
python -m balancemm --model BaseClassifier --trainer OGMTrainer --lr 0.003 --alpha 0.2 --dataset Mosei --device 0
python -m balancemm --model BaseClassifier --trainer GBlendingTrainer --lr 0.003 --alpha 1.0 --dataset Mosei --device 0
python -m balancemm --model BaseClassifier_AMCo --trainer AMCoTrainer --lr 0.003 --alpha 1.0 --dataset Mosei --device 1
python -m balancemm --model BaseClassifier --trainer baselineTrainer --lr 0.01 --alpha 1.0 --dataset Mosei --device 1
python -m balancemm --model BaseClassifier --trainer GBlendingTrainer --lr 0.01 --alpha 1.0 --dataset Mosei --device 0


python -m balancemm --model BaseClassifier --trainer baselineTrainer --lr 0.01 --alpha 1.0 --dataset Mosei --device 1 --mode all_test

python -m balancemm --model BaseClassifier --trainer OGMTrainer --lr 0.01 --alpha 0.3 --mu 0 --scaling 0 --lam 0 --eta 0 --dataset Mosei --device 0 --mode all_test
python -m balancemm --model BaseClassifier --trainer unimodalTrainer --lr 0.01 --alpha 1.0 --dataset Mosei --device 1