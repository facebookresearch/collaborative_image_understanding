# Collaborative Image Understanding

## Abstract
Automatically understanding the contents of an image is a highly relevant problem in practice. In e-commerce and social media settings, for example, a common problem is to automatically categorize user-provided pictures. Nowadays, a standard approach is to fine-tune pre-trained image models with application-specific data. Besides images, organizations however often also collect \emph{collaborative} signals in the context of their application, in particular how users interacted with the provided online content, e.g., in forms of viewing, rating, or tagging. Such signals are commonly used for item recommendation, typically by deriving \emph{latent} user and item representations from the data. In this work, we show that such collaborative information can be leveraged to improve the classification process
of new images. Specifically, we propose a multitask learning framework, where the auxiliary task is to reconstruct collaborative latent item representations. A series of experiments on datasets from e-commerce and social media demonstrates that considering collaborative signals helps to significantly improve the performance of the main task of image classification by up to 9.1\%.

## Download data
Pinterest: https://nms.kcl.ac.uk/netsys/datasets/social-curation/dataset.html

MovieLens: https://www.kaggle.com/ghrzarea/movielens-20m-posters-for-machine-learning

Amazon product data (Clothing/Toys): http://jmcauley.ucsd.edu/data/amazon/

## Run experiments
To replicate the experiments for the Pinterst dataset

```
cd bash_script

# Train recommender
./2_1_pinterest_initial.sh

# Optimize for the best cf weight
./2_2_Pinterest_cf_weight.sh

# Train baselines
./2_3_Pinterest_baselines.sh

# Train with different label ratio
./2_4_Pinterest_label_ratio

```

Follow the other files in the bash_script folder for the other datasets. Then update result file:
```
/home/ubuntu/cactus/outputs/pinterest/results.yaml
```
and run 
```
python main_predict_testset.py dataset_name=pinterest
```
Lastly to produce the figures run
```
python main_evaluate_methods.py 
```

# Citing
If you use this code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{bibas2021single,
  title={Collaborative Image Understanding},
  author={Bibas, Koby and Sar Shalom, Oren and Jannach, Dietmar},
  booktitle={The 31th {ACM} International Conference on Information
               and Knowledge Management, 2022},,
  year={2022}
}
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This repository is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licensed, as found in the [LICENSE](LICENSE) file.

