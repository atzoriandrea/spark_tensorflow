import os
import numpy as np
import tensorflow as tf
from pyspark.sql.types import DoubleType, ArrayType, FloatType
import pandas as pd
import subprocess
from pyspark.sql.functions import  col, pandas_udf, PandasUDFType
from pyspark.sql import SparkSession

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" # Must corrispond to the current jdk used by colab
os.environ["SPARK_HOME"] = "/opt/spark-3.0.1-bin-hadoop2.7/" # Must corrispond with the downloaded spark (1st line)
spark = SparkSession.builder.master("spark://172.31.0.101:7077").appName("distributedPrediction").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("Error")

def run_cmd(args_list):
        print('Running system command: {0}'.format(' '.join(args_list)))
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s_output, s_err = proc.communicate()
        s_return =  proc.returncode
        return s_return, s_output, s_err

def get_model():
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

def make_test_set():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test[..., tf.newaxis], y_test[..., tf.newaxis]

def parse_image(image_data):
  image = tf.reshape(image_data,[28,28,1])
  return image

print("Acquiring trained model...")
model = tf.keras.models.load_model("./trained_model.h5")
print("Broadcasting model weights...")
bc_model_weights = sc.broadcast(model.get_weights())

@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
def predict_batch_udf(image_batch_iter):
  batch_size = 1
  model = get_model()
  model.set_weights(bc_model_weights.value)
  for image_batch in image_batch_iter:
    images = np.vstack(image_batch)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(parse_image, num_parallel_calls=8).prefetch(5000).batch(batch_size)
    preds = model.predict(dataset)
    yield pd.Series(list(preds))

def compute_accuracy(y_true, y_pred):
    preds = y_pred.reshape(-1,1)
    acc = (sum(preds==y_true)/len(y_pred))*100
    print("Accuracy: " + str(acc[0]) + "%")


def main():
    files, labels = make_test_set()
    file_name = "image_dataMNIST.parquet"

    image_data = []
    label_data = []
    for file, label in zip(files, labels):
        data = np.asarray(file, dtype="float32").reshape([28 * 28 * 1])
        image_data.append(data)
        label_data.append(np.array(label, dtype="float32"))
    df = {'data': image_data,'y_true': label_data}
    pandas_df = pd.DataFrame(df)
    print("Creating dataset parquet file...")
    pandas_df.to_parquet(file_name)
    print("Copying dataset parquet file into HDFS...")
    (ret, out, err) = run_cmd(['hdfs', 'dfs', '-copyFromLocal', file_name, "/user/ubuntu/"])
    df = spark.read.parquet(file_name)
    print("Found: " + str(df.count()) + " images into HDFS parquet file")
    print("Computing predictions...")
    output_file_path = "./predictions"
    predictions_df = df.select(predict_batch_udf(col("data")).alias("prediction"))
    predictions_df.write.mode("overwrite").parquet(output_file_path)

    result_df = spark.read.load(output_file_path)
    result_df = result_df.toPandas().to_numpy()
    predictions = np.asarray([x.index(max(x)) for x in result_df.flatten().tolist()])

    compute_accuracy(labels, predictions)

if __name__ == "__main__":
    main()
