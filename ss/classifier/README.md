This repo contain source code for AICR project (Classification model)

## generate data	
Use script **generate_data.py** to generate data for classifier1 and classifier2
- Use function "gen_data_classifier1_from_detector" to gen data for classifier1
- Use function "gen_data_classifier2_from_detector" to gen data for classifier2

## train/test classifier1 (3 stages)
  - To train classifier1, run command `python train_classifier1.py`
  - To test classifier1, run command `python predict_classifier1.py`
  - To modify some parameters for train/test, use config file **config/classifier_config.ini**

## train/test classifier2 (3 stages)
  - To train classifier2, run command `python train_classifier2.py`
  - To test classifier2, run command `python predict_classifier2.py`
  - To modify some parameters for train/test, use config file **config/classifier_config.ini**
  
## train/test classifier12 (2 stages)
  - To train classifier12, run command `python train_classifier12.py`
  - To test classifier12, run command `python predict_classifier12.py`
  - To modify some parameters for train/test, use config file **config/classifier_config.ini**
