
from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark.sql import SparkSession
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" # Must corrispond to the current jdk used by colab
os.environ["SPARK_HOME"] = "/opt/spark/"
# Adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow as tf
    import uuid

    def make_datasets():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        dataset = tf.data.Dataset.from_tensor_slices((x_train[..., tf.newaxis], y_train[..., tf.newaxis])).batch(1)
        #dataset = dataset.repeat().batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model2 = tf.keras.models.Sequential()
        model2.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                          input_shape=(28, 28, 1)))
        model2.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model2.add(tf.keras.layers.Flatten())
        model2.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model2.add(tf.keras.layers.Dense(10, activation='softmax'))
        model2.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'],
        )
        return model2

    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=10, steps_per_epoch=60000//(32*2))


spark = SparkSession.builder.master("spark://192.168.1.38:7077").appName("testTrain")\
    .config("spark.driver.memory" , "4g").\
    config("spark.executor.memory" , "4g").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("Error")
MirroredStrategyRunner(num_slots=20, spark=spark, use_gpu=False).run(train)
