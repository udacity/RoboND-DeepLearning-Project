# AWS Setup for RoboND-DeepLearning Project

## Create an AWS account
- Instructions can be found [here](https://classroom.udacity.com/nanodegrees/nd209-beta/parts/4405411f-bdc8-43fb-9e2b-b4b98a61c760/modules/375d6b1e-b31a-4d2e-b33a-996167faf77e/lessons/dca6b1cd-5d40-41ab-a502-69a24c4b227e/concepts/bc84756d-22ba-448d-bad5-a63f99b3c693?contentVersion=1.0.0&contentLocale=en-us)

## Request a Limit increase
- Instructions can be found [here](https://classroom.udacity.com/nanodegrees/nd209-beta/parts/4405411f-bdc8-43fb-9e2b-b4b98a61c760/modules/375d6b1e-b31a-4d2e-b33a-996167faf77e/lessons/dca6b1cd-5d40-41ab-a502-69a24c4b227e/concepts/bc84756d-22ba-448d-bad5-a63f99b3c693?contentVersion=1.0.0&contentLocale=en-us)

## Create and Launch an EC2 instance
1. Open your [AWS Console](https://console.aws.amazon.com/console/)
2. Next select **Services > Compute > EC2**

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/1.png)

3. To start using Amazon EC2, click the **Launch Instance** button

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/2.png)

4. Next you will select an AMI. AMI or Amazon Machine Image is a template that contains the software configuration (operating system, application server, and applications) required to launch your instance:
   - Select community AMIs from the left side panel
   
   ![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/3.png)
   
   - In the search box enter **Udacity RoboND**
   - Click select for the AMI: ***Udacity-RoboND-Deep-Learning-Laboratory***
   
   ![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/4.png)

5. Choose Instance type: Select a **p2.xlarge** instance

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/5.png)

6. Configure Instance Details: These details can be left the same, click **Next:Add Storage**

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/6.png)

7. Add Storage: If you would like to not have your data deleted between runs you can deselect **Delete on Termination** but keep in mind, this could potentially incur a small credit usage over time.

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/7.png)

8. Add Tags: No changes to be made here

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/8.png)

9. Configure Security Group: Select **Create a new security group** and from the drop down box under **Source** select **My IP**

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/9.png)

10. Click **Review and Launch** button in the lower-right corner 
11. On the review page click **Launch** in the lower-right corner

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/10.png)

12. Click the check box to select key pair, if you have not created a key pair you can create a new key pair right here

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/11.png)

13. Finally click **Launch instances**
14. If you get an error saying you do not have any instances of the chosen type available, make sure your Limit Increase request has been approved.
15. You may click **View Instances** to see the instance you just launched

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/12.png)

## Connecting to an EC2 Instance
1. At anytime you can access your **EC2 Dashboard** by selecting **Services > EC2**

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/16.png)

2. To open a list of your EC2 instances Select **Instances** from the left side panel.
3. Now, select your instance from the Instances table and click the **Connect** button at the top of the screen.

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/13.png)

4. Open a new terminal window on your system.
5. `$cd` to the location where you have stored your private key and change it's permissions
    ```sh
    $ chmod 400 your_key.pem 
    ```
6. Amazon provides you the ssh command to connect to your instance using its Public DNS. But you need to modify that command such that you are logging in as **ubuntu** and not **root**. For example if the suggestion from amazon is:
    ```sh
    $ ssh -i "first_key_pair.pem" root@ec2-52-43-242-129.us-west-2.compute.amazonaws.com
    ```
    you must change it to:
    ```sh
    $ ssh -i "first_key_pair.pem" ubuntu@ec2-52-43-242-129.us-west-2.compute.amazonaws.com
    ```
7. **Note:** Do not copy the above line into your terminal directly, copy the line in Connect to Your Instance dialog with above stated modification.

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/14.png)

8. Type yes when asked if you want to connect. You should now have access to your EC2 instance within your active terminal.

## Disconnecting and Stopping an instance
- To simply disconnect from a connected instance. Type `$exit` in your active terminal.
- If you are no longer using an instance and want to stop it temporarily, select your instance from the list then click **Actions > Instance State > Stop**

![](https://github.com/udacity/RoboND-DeepLearning-Private/blob/master/docs/misc/AWS%20setup%20images/15.png)

- Remember, when you stop an instance, it is shut down by AWS. You will not be charged hourly usage for a stopped instance but you will be charged for any Amazon EBS storage volumes. Each time you start a stopped instance you will be charged a full instance hour, even if you make this transition multiple times within a single hour.
- More information on Starting & Stopping an instance can be found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html)

## TODO:

## Training the network on your instance
1. Check `run.py` and make sure it is configured correctly. Refer [this]() document.
2. Train your network by using:
  ```sh
  $ python run.py
  ```
  
## Attaching, preparing and mounting an ebs volume
1. Navigate to volumes in the EC2 side panel
2. Create an EBS volume
3. Once an instance is running click on actions and select attach, then select the name of the running instance
4. If it is the first time using the volume you will have to format the volume.
5. To mount the volume:
  ```sh
  $ sudo mount /dev/xvdf /home/ubuntu/ebs_vol
  ```
6. Change the ownership of EBS volume:
```
$ sudo chown -R ubuntu ebs_vol
$ mkdir ebs_vol/dl_files
$ scp -r -i ~/first_key_pair.pem RoboND-DeepLearning-Private_aws/ ubuntu@ec2-52-38-165-179.us-west-2.compute.amazonaws.com:~/ebs_vol/dl_files
$ pip install tqdm
```
