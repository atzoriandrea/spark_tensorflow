# Spark_Tensorflow

Distribute tensorflow training and predictions through spark cluster

This source code is useful for working with tensorflow in a distributed mode using Apache Spark.

We used these packages: \
Spark 3.0.1\
Hadoop 2.7.7 or more recent versions\
numpy 1.18.5\
tensorflow 2.3.0\
pyspark 3.0.1\
pandas 1.1.5\
spark_tensorflow_distributor 0.1.0\
matplotlib 3.3.3

### In order to install project requirements (listed above)
```
pip3 install -r requirements.txt
```

### Note
We assume that you already have configured your spark and hadoop environments. If you did not, configure them before proceeding.

---

### Project description

This project has been developed using the MNIST dataset, since it is the most known one and is already included in tensorflow package.
You will need to modify both *distributed_training.py* and *distributed_prediction_and_test.py* in order to make them working with your own spark and hadoop environment.

---
#### distributed_training.py

In this file, we acquire the MNIST train set. Then, we preprocess it in order to make it distributable on cluster nodes.\
Then, each node will create his own model and all the training steps are performed in synchronized mode on the entire cluster, using ALL AVAILABLE CPU CORES.
Then, the trained model will be saved on the driver node.

---

#### distributed_prediction_and_test.py

In this file, we acquire the MNIST test set. Then, we distribute a .parquet file into HDFS in order to make it available to all nodes.
Them, the framework will compute all the predictions and the resulting accuracy score.

---
#### Spark_Tensorflow_distributed_Plot.ipynb

In this file, simply we create a plot in order to represent differences in training and prediction times with different number of nodes.


###### Example
![plot](./plot.png)

