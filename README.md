# Code execution on Amazon AWS using Terraform
---

#### 0 Terraform Download and Installation

```
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=$(dpkg --print-architecture)] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt install terraform
```

#### 1 AWS instances configuration by Terraform
```
mkdir distributed_project
cd distributed_project
git clone https://github.com/DanyOrtu97/Spark-Terraform-.git
mv Spark-Terraform- spark-terraform-master
mkdir spark-terraform-master/app
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
You can create the subnet_id on your AWS account in EC2 > Network interfaces > Create a network interface and choose the subnet for "us-east-1c" region.
After the creation you can put it in the rows described above

##### NOTE: If the security group "Hadoop_cluster_sc" on EC2 > Security Group is already in your AWS account, you must delete it in order to avoid duplicate error messages 

Make sure that the zone/region just choice ("us-east-1c") is the same of "variable.tf" file into spark-terraform-master folder already downloaded
```
variable "region" {
    type = string
    default = "us-east-1"
}
```

and in "terraform.tfstate" file into spark-terraform-master folder
```
"availability_zone": "us-east-1c",
```

You need to check and eventually correct the region in every place in which appear, with your region "us-east-1c"


Now in the opened terminal on "spark-terraform-master" folder:

```
ssh-keygen -f localkey
```

Login to your AWS account and create a new key pairs with name "amzkey" and in ".pem" format.
You can follow the guide on https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair if you have problems with key generation.
Download amzkey.pem and copy it into your spark-terraform-master folder

```
chmod 400 amzkey.pem  //change permissions for the key
terraform init 
terraform apply
```
Type 'yes' when requested

##### NOTE: The process can take some time, due to the installations and operations it has to perform on each node 

#### 2 Connection to the master instance on AWS by amzkey.pem
At this point, you must have all your instances created on AWS.
Open a terminal on the spark-terraform-master directory where there is the key file and type:
```
ssh -i amzkey.pem ubuntu@[address of master instance]
```
The master is named with hostname s01

#### 3 Run Spark and Hadoop on the master
In the same terminal where you connecting to the master instance, you need to type this command in order to starting Spark and Hadoop:
```
sh spark-start-master.sh
```

```
sh hadoop-start-master.sh
```

#### 4 Connection to slaves instances
Now one by one you must connect with the slaves using the command:
```
ssh [name slave]  //Example ssh s02
```

##### NOTE: You can find the slaves' hostnames on "variable.tf" file inside spark-terraform-master folder
```
variable "hostnames" {
    default = {
        "0" = "s02"
        "1" = "s03"
    }
```

#### 5 Run the slave instances
On each slave instance you can run this command in order to active the instance for the computation:
```
sh spark-start-slave.sh
```

#### 6 Modify the number of cores to run in spark_tensorflow/distributed_training.py
If it is necessary you can modify the number of cores that the nodes on the cluster can use. From the master terminal open the file distributed_training.py and change this row (row 64) with the number of cores. 
Note that we have 8 slave nodes, each of these have 2 cores, so if you start the training with 2 nodes, you should use 4 cores.
```
weights = MirroredStrategyRunner(num_slots=4, spark=spark, use_gpu=False).run(train) //num_slots represent the number of cores
```

#### 7 Modify spark_tensorflow_distributor package  
You must add two rows on the spark_distributor package in order to pass the spark session at this function with spark = spark:
```
weights = MirroredStrategyRunner(num_slots=sc.defaultParallelism, spark=spark, use_gpu=False).run(train)
```
You need to add a row in "mirror_strategy_runner.py" in the def at line 52:

You can modify this file by executing the following row in master's terminal
```
sudo nano /home/ubuntu/.local/lib/python3.8/site-packages/spark_tensorflow_distributor/mirrored_strategy_runner.py
```
then
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

#### 8 Modify the number of epochs (optional) in spark_tensorflow/distributed_training.py
You can modify the number of epochs on file distribute_training.py at row 56:
```
multi_worker_model.fit(x=train_datasets, epochs=1, steps_per_epoch=60000//32)
```

#### 9 From the master' terminal run the distribute_training.py using this command:
```
python3 spark_tensorflow/distribute_training.py
```
During the training step you can control on the Spark GUI on the browser 
After the training step yuo have a model saved on the hadoop cluster and you can run the prediction code

#### 10 From the master ' terminal run distribute_prediction_and_test.py using this command:
```
python3 spark_tensorflow/distribute_prediction_and_test.py
```
You have finished the computation and you can modify the number of nodes of the cluster in order to test with a different situation.
