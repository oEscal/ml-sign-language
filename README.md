# ml-sign-language

## Dataset  
 - Since the original dataset (`sign_mnist_test.csv + sign_mnist_train.csv`) have 34627 entries, we decided to let the `sign_mnist_test.csv` dataset with the original data and split the `sign_mnist_train.csv` into 2 datasets, one with the training data (`sign_mnist_train.csv`) with 20456 entries and another with the cross validation data (`sign_mnist_cv.csv`), with 7000 entries, giving a division of more or less:
 | Trainig set | Cross validation set | Test set |
 |-------------|----------------------|----------|
 |     60%     |          20%         |    20%   |
