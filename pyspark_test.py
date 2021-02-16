import time
import os
from pyspark.sql import SparkSession
from spark_tensorflow_distributor import MirroredStrategyRunner
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" # Must corrispond to the current jdk used by colab
os.environ["SPARK_HOME"] = "/opt/spark/"
 # Must corrispond with the downloaded spark (1st line)

def random_shuffled_dataset_part(preprocess):
    import sklearn
    import glob
    import cv2
    import numpy as np
    directory = preprocess.directory
    train_full_paths = []
    normals = glob.glob(directory+"/NORMAL/*.jpeg")
    pneumo = glob.glob(directory+"/PNEUMONIA/*.jpeg")
    train_full_paths = [cv2.imread(file) for file in normals[0:7]]
    train_full_paths.extend([cv2.imread(file) for file in pneumo[0:7]])
    train_set = []
    for image in train_full_paths:
        train_set.append(cv2.resize(image,(320,320)))
    zeros = [0] * 7
    ones = [1] * 7
    labels = zeros+ones #preprocess.labels.tolist()
    a,b = sklearn.utils.shuffle(train_set, labels)
    return np.array(a), np.array(b)

def random_shuffled_dataset(preprocess):
    import sklearn
    import glob
    import cv2
    import numpy as np
    directory = preprocess.directory
    train_full_paths = []
    train_full_paths = [cv2.imread(file) for file in glob.glob(directory+"/NORMAL/*.jpeg")]
    train_full_paths.extend([cv2.imread(file) for file in glob.glob(directory+"/PNEUMONIA/*.jpeg")])
    train_set = []
    for image in train_full_paths:
        train_set.append(cv2.resize(image,(320,320)))
    labels = preprocess.labels.tolist()
    a,b = sklearn.utils.shuffle(train_set, labels)
    return np.array(a), np.array(b)


def training():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0}
    )

    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session = tf.compat.v1.Session(config=config)
    import uuid
    BUFFER_SIZE = 10000
    BATCH_SIZE = 8

    #ACQUIRING DATASET AND PREPARING
    print("Spark application started...")
    train_dir = "/media/andrea/Dati2/chest_xray/chest_xray/train"
    test_dir = "/media/andrea/Dati2/chest_xray/chest_xray/test"
    val_dir = "/media/andrea/Dati2/chest_xray/chest_xray/val"

    print("Train set:\n========================================")
    num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    print(f"PNEUMONIA={num_pneumonia}")
    print(f"NORMAL={num_normal}")

    print("Test set:\n========================================")
    print(f"PNEUMONIA={len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}")
    print(f"NORMAL={len(os.listdir(os.path.join(test_dir, 'NORMAL')))}")

    print("Validation set:\n========================================")
    print(f"PNEUMONIA={len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))}")
    print(f"NORMAL={len(os.listdir(os.path.join(val_dir, 'NORMAL')))}")

    image_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    batch_size = 8
    img_dim = 320
    classes = 2


    train = image_generator.flow_from_directory(train_dir,
                                                batch_size=8,
                                                shuffle=True,
                                                class_mode='binary',
                                                target_size=(320, 320))


    validation = image_generator.flow_from_directory(val_dir,
                                                     batch_size=8,
                                                     shuffle=True,
                                                     class_mode='categorical',
                                                     target_size=(320, 320))

    test = image_generator.flow_from_directory(test_dir,
                                               batch_size=8,
                                               shuffle=True,
                                               class_mode='binary',
                                               target_size=(320, 320))
    ds = tf.data.Dataset.from_generator(
        lambda: test,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, img_dim, img_dim, 3],
                       [batch_size, classes])
    )
    print("Acquiring and shuffling dataset")

    weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)
    weight_for_1 = num_normal / (num_normal + num_pneumonia)
    class_weigths = {0: weight_for_0, 1: weight_for_1}

    def build_and_compile_cnn_model():
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(320, 320, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(320, 320, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    #train_dataset = tf.data.Dataset.from_tensor_slices((shuffle_train, shuffle_labels_train)).batch(BATCH_SIZE)
    #val_dataset = tf.data.Dataset.from_tensor_slices((shuffle_valid, shuffle_labels_valid)).batch(len(shuffle_valid))

    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    #ds = ds.with_options(options)
    #train_dataset = train_dataset.with_options(options)

    print("Creating model")
    multi_worker_model = build_and_compile_cnn_model()
    print("Model Created!")
    print("Training...")
    start_time = time.time()
    #multi_worker_model.fit(train_dataset, validation_data=val_dataset,epochs=5, batch_size=BATCH_SIZE, class_weight=class_weigths, verbose=True)
    multi_worker_model.fit(ds, epochs = 5, batch_size=BATCH_SIZE, class_weight = class_weigths, verbose=True)

    print("--- %s seconds ---" % (time.time() - start_time))

spark = SparkSession.builder.master("spark://192.168.1.38:7077").appName("testTrain")\
    .config("spark.driver.memory" , "2g").\
    config("spark.executor.memory" , "2g").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("Error")

MirroredStrategyRunner(num_slots=2,use_gpu=False,spark=spark, local_mode=False,use_custom_strategy=False).run(training)