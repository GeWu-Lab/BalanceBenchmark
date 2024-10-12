#!/bin/bash
# lr = 0.01
python -m balancemm --Main_config.Trainer OGMTrainer --Main_config.device 2  --Main_config.dataset Cremad --Train.dataset Cremad --Test.dataset Cremad --train.parameter.base_lr 0.01