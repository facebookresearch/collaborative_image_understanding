cd ../src
python main_train_model.py -m category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=0.0,0.5,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5