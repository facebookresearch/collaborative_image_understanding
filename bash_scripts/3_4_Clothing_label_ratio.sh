cd ../src
python main_train_model.py -m category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=2.0 \
    labeled_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

python main_train_model.py -m category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=0.0 \
    labeled_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0