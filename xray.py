import os
#import findspark
import pyspark
#import pydoop.hdfs as hdfs

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
from pyspark.sql import SparkSession

#findspark.init()



os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" # Must corrispond to the current jdk used by colab
os.environ["SPARK_HOME"] = "/opt/spark/" # Must corrispond with the downloaded spark (1st line)
#sc = pyspark.SparkContext(master="spark://192.168.1.38:7077", appName="GGG", conf=conf).getOrCreate()
spark = SparkSession.builder.master("spark://192.168.1.38:7077").\
    appName("testTrain").\
    enableHiveSupport().\
    config("spark.driver.resource.gpu.discoveryScript", "/opt/spark/examples/src/main/scripts/getGpusResources.sh").\
    config("spark.driver.resource.gpu.amount", "1").\
    getOrCreate()
#spark.conf.set("spark.driver.resource.gpu.amount", "1")
#spark.conf.set("spark.task.resource.gpu.amount", "1")
#sc.hadoopConfiguration.hconf.setInt("dfs.replication", 2)
sc = spark.sparkContext
sc.setLogLevel("Error")


words = sc.parallelize(['the','and','you','then','what','when','steve','where','savannah','research'])
data = words.map(lambda x:(len(x),(x,1)))
grouped = data.reduceByKey(lambda x,y:(x[0]+' '+y[0],x[1]+y[1]))
print(grouped.collect())