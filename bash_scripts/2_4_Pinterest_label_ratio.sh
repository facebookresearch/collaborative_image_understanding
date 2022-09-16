cd ../src
python main_train_model.py -m category=pinterest \
    arch=regnet \
    batch_size=128 \
    lr=0.1 \
    milestones=[10,14] \
    epochs=15 \
    cf_weight=5.0 \
    labeled_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

python main_train_model.py -m category=pinterest \
    arch=regnet \
    batch_size=128 \
    lr=0.1 \
    milestones=[10,14] \
    epochs=15 \
    cf_weight=0.0 \
    labeled_ratio=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
