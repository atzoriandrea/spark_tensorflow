# Code execution on Amazon AWS using Terraform
 
#### 0 Terraform Download and Installation

```
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=$(dpkg --print-architecture)] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt install terraform
```

#### 1 AWS istances configuration by Terraform
```
mkdir distributed_project
cd distributed_project
git clone https://github.com/DanyOrtu97/Spark-Terraform-.git
cd spark-terraform-master/app
git clone https://github.com/atzoriandrea/spark_tensorflow.git
cd ..
```
Update terraform.tfvars file
```
access_key="<YOUR AWS ACCESS KEY>"
secret_key="<YOUR AWS SECRET KEY>"
token="<YOUR AWS TOKEN>"
```
You can find the AWS ACCES KEY, AWS SECRET KEY and AWS TOKEN on your AWS panel account on "Account Details"


Modify the subnet_id on "main.tf" file at row 41 and 109 into spark-terraform-master folder already downloaded

for namenode at row 41
```
# namenode (master)
resource "aws_instance" "Namenode" {
  subnet_id = "subnet-1c40f47a"
  count         = var.namenode_count
  ami           = var.ami_image
  instance_type = var.instance_type
  key_name      = var.aws_key_name
  tags = {
    Name = "s01"
  }
  private_ip             = "172.31.0.101"
  vpc_security_group_ids = [aws_security_group.Hadoop_cluster_sc.id]
```

for datanodes at row 109
```
# datanode (slaves)
resource "aws_instance" "Datanode" {
  subnet_id = "subnet-1c40f47a"
  count         = var.datanode_count
  ami           = var.ami_image
  instance_type = var.instance_type
  key_name      = var.aws_key_name
  tags = {
    Name = lookup(var.hostnames, count.index)
  }
  private_ip             = lookup(var.ips, count.index)
  vpc_security_group_ids = [aws_security_group.Hadoop_cluster_sc.id]
```
You can find the subnet_id on your AWS account searching in the master's info above subnet_id information


Make sure that the zone/region on your AWS instances is the same of "variable.tf" file into spark-terraform-master folder already downloaded
```
variable "region" {
    type = string
    default = "us-east-1"
}
```

The same as in the "variable.tf" file is needed in "terraform.tfstate" file into spark-terraform-master folder
```
"availability_zone": "us-east-1d",
```

You need to adapt the region in every place in which appear, with your region


```
ssh-keygen -f localkey
```
Login to your AWS account and create a new key pairs with name "amzkey" and in ".pem" format.
You can follow the guide on https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair if you have problems with key generation.
Download amzkey.pem and copy it into your spark-terraform-master folder

```
chmod 400 amzkey.pem
terraform init 
terraform apply
```
Type 'yes' when requested

#### 2 Connection to the master istance on AWS by amzkey.pem
At this point, you must have all your instances running on AWS.
Open a terminal on the spark-terraform-master directory where there is the key file and type:
```
ssh -i amzkey.pem ubuntu@[address of master instance]
```
Yuo can find the master' address on the AWS console in the instance informations

#### 3 Run Spark and Hadoop on the master
In the same terminal where you connecting to the master instance, you need to type this command in order to starting Spark and Hadoop:
```
sh spark-start-master.sh
```

```
sh hadoop-start-master.sh
```

#### 4 Connection to slaves istances
Now one by one you must connect with the slaves using the command:
```
ssh [name slave]  //Example ssh s02
```

#### 5 Run the slave istances
On each slave instance you can run this command in order to active the instance for the computation:
```
sh spark-start-slave.sh
```

#### 6 Change the Java environment on spark_tensorflow-master/distribute_training.py
Open the project folder of distributed training on spark, already downloaded and open the file distribute_training.py inside spark_tensorflow-master folder
```
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" //Java home environment path
```

#### 7 Change the Spark Home path
Open the project folder of distributed training on spark, already downloaded and open the file distribute_training.py inside spark_tensorflow-master folder
```
os.environ["SPARK_HOME"] = "/opt/spark-3.0.1-bin-hadoop2.7/"
```

#### 8 Modify the number of cores to run
If it is necessary you can modify the number of cores that the nodes on the cluster can use. From the master terminal open the file distributed_training.py and change this row (row 64) with the number of cores. 
Note that we have 8 slave nodes, each of these have 2 cores, so if you start the training with 2 nodes, you should use 4 cores.
```
weights = MirroredStrategyRunner(num_slots=4, spark=spark, use_gpu=False).run(train) //num_slots represent the number of cores
```

#### 9 Modify spark_tensorflow_distributor package  
You must add two rows on the spark_distributor package in order to pass the spark session at this function with spark = spark:
```
weights = MirroredStrategyRunner(num_slots=sc.defaultParallelism, spark=spark, use_gpu=False).run(train)
```
You need to add a row in "mirror_strategy_runner.py" in the def at line 52:

```
def __init__(self,
                 *,
                 num_slots,
                 local_mode=False,
                 spark = None,
                 use_gpu=True,
                 gpu_resource_name='gpu',
                 use_custom_strategy=False):
```

and to change the row 135 inserting:
```
if spark is None:
  spark = SparkSession.builder.getOrCreate()
```

#### 10 Modify the number of epochs (optional)
You can modify the number of epochs on file distribute_training.py at row 56:
```
multi_worker_model.fit(x=train_datasets, epochs=1, steps_per_epoch=60000//32)
```

#### 11 From the master' terminal run the distribute_training.py using this command:
```
python3 distribute_training.py
```
During the training step you can control on the Spark GUI on the browser 
After the training step yuo have a model saved on the hadoop cluster and you can run the prediction code

#### 12 From the master ' terminal run distribute_prediction_and_test.py using this command:
```
python3 distribute_prediction_and_test.py
```
You have finished the computation and you can modify the number of nodes of the cluster in order to test with a different situation.
