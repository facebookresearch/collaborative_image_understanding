cd ../src
python main_train_recommender.py category=Toys_and_Games test_size=0.0 models=[]
python main_process_labels.py category=Toys_and_Games
python main_train_model_cf_based.py category=Toys_and_Games save_as_asset=true