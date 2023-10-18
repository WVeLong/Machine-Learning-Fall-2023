from pathlib import Path
import os

parent_directory = os.path.dirname(os.getcwd())
Train_1_data_path = Path(parent_directory, 'data', 'train1_icu_data.csv')
Train_2_data_path = Path(parent_directory, 'data', 'train2_icu_data.csv')
Train_1_label_path = Path(parent_directory, 'data', 'train1_icu_label.csv')
Train_2_label_path = Path(parent_directory, 'data', 'train2_icu_label.csv')
Test_1_data_path = Path(parent_directory, 'data', 'test1_icu_data.csv')
Test_2_data_path = Path(parent_directory, 'data', 'test2_icu_data.csv')
Test_1_label_path = Path(parent_directory, 'data', 'test1_icu_label.csv')
Test_2_label_path = Path(parent_directory, 'data', 'test2_icu_label.csv')
