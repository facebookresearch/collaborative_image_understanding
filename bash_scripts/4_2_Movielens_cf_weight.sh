cd ../src
python main_train_model.py -m category=movielens \
    arch=mobilenet \
    batch_size=128 \
    lr=0.01 \
    milestones=[40] \
    epochs=45 \
    cf_weight=0.0,0.5,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5