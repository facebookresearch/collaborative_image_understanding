cd ../src
python main_train_model.py -m category=pinterest \
    arch=regnet \
    batch_size=128 \
    lr=0.1 \
    milestones=[10,18] \
    epochs=20 \
    cf_weight=0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5