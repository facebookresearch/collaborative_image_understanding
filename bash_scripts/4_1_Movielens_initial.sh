cd ../src
python main_train_recommender.py data_dir=/data/movielens_dataset category=movielens test_size=0.0 models=[]
python main_process_labels_movielens.py
python main_train_model_cf_based.py category=movielens save_as_asset=true