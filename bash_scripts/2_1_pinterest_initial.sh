cd ../src
python main_train_recommender.py data_dir=/data/pinterest_dataset category=pinterest test_size=0.0 models=[]
python main_train_model_cf_based.py category=pinterest save_as_asset=true