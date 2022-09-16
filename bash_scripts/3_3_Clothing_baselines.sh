cd ../src
python train_model_with_cf_pretraining.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=512 \
    epochs=100

python main_train_model.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=2.0 \
    confidence_type=pos_label_loss_based \
    cf_loss_type=exp \
    conf_max_min_ratio=3.0

python main_train_model.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=256 \
    cf_weight=2.0 \
    confidence_type=num_intercations \
    cf_loss_type=exp \
    conf_max_min_ratio=3.0

python main_train_model.py category=Clothing_Shoes_and_Jewelry \
    arch=resnet18 \
    batch_size=128 \
    cf_weight=2.0 \
    cf_loss_type=triplet
