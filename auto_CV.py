# from dataquality.dq_auto.image_classification import auto
import dataquality as dq

# dq.auto(train_data="/Users/bogdan/Code/DATA/CIFAR-10-images-master/train_all", test_data="/Users/bogdan/Code/DATA/CIFAR-10-images-master/test_all")
dq.auto(hf_data="beans")
