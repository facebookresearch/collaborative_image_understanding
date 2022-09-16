cd ../src
python main_train_recommender.py category=Clothing_Shoes_and_Jewelry test_size=0.0 models=[]
python main_process_labels.py category=Clothing_Shoes_and_Jewelry num_samples_threshold=20
python main_train_model_cf_based.py category=Clothing_Shoes_and_Jewelry save_as_asset=true