# fedmaldetect

---

## Local Simulation
### Step 1: Clone the Repository
```bash
git clone https://github.com/secure-dotcom/fedmaldetect.git
cd fedmaldetect
```
### Create a Python virtual environment and install dependencies.
```bash
  python3 -m venv project
  source project/bin/activate
  pip install -r requirements.txt
```
### Download the dataset and save it in a directory named after the dataset
1. [iot-23](https://drive.google.com/file/d/1Nh8-FrUSNwJ4SgWsFbAZkvA4x7SEF2fY/view?usp=sharing)
2. [radar](https://drive.google.com/file/d/1JtZttnbDnWgm1HcNOV1AY_H8iXExpsUR/view?usp=sharing)
3. [nbaiot](https://drive.google.com/file/d/1UnWdDrp5rOWmqqedjRqTCBrJiDQz4FA7/view?usp=sharing)

- Copy the paths of all datasets and insert them into lines 138, 140, and 142, respectively in semisupervised.py, according to their corresponding dataset names.

### To run local simulation for centralized and decentralized comparison with nbaiot dataset (Results of Table 5 in the paper):
```bash
    python semisupervised.py --dataset=nbaiot --num-clients=10 --training_type=Decentralized
    python semisupervised.py --dataset=nbaiot --num-clients=10 --training_type=Centralized
  
```
### To run local simulation for 10 clients nbaiot dataset (Results of Fig. 6 in the paper):
```bash
    python semisupervised.py --dataset=nbaiot --num-clients=10 --training_type=Decentralized
```
Note: to vary clients in nbaiot dataset, you can change --num-clients value from 2 to 49 (Results of Fig. 7 in the paper)

### To run local simulation for 10 clients with different datasets (Results of Table 6 in the paper):
```bash
    python semisupervised.py --dataset=nbaiot --num-clients=10
    python semisupervised.py --dataset=radar --num-clients=10
    python semisupervised.py --dataset=iot23 --num-clients=10
```

### To check PR-AUC and ROC-AUC scores vs Communication rounds (Results of Fig. 8 and 9 in the paper):
```bash
    python semisupervised.py --dataset=nbaiot --num-clients=10 --training_type=Decentralized --save=True
    python semisupervised.py --dataset=nbaiot --num-clients=10 --training_type=Decentralized --save=True --aggregation=FedAvg
```

### Results for calculating FLOPs, MACs and No of Params is displayed on any command running semisupervised.py (Results in Table 7 in the paper):
```bash
    python semisupervised.py
```

## IoT Testbed Simulation
### Step 1: Clone the Repository
```bash
git clone https://github.com/secure-dotcom/fedmaldetect.git
cd fedmaldetect
```
### Step 2: Set up Multiple Raspberry PI client devices
- Use weak password and username combinations in some Raspberry Pi client devices (e.g., admin:admin) and enable the telnet port in all these devices to make them vulnerable. 
- Follow the steps below in all Raspberry Pi devices.
- Copy the requirements.txt and the pyproject.toml inside the embedded-devices directory to all Raspberry PI clients' devices.
- Create a Python virtual environment and install dependencies.
  ```bash
  python3 -m venv project
  source project/bin/activate
  pip install -e .
  pip install -r requirements.txt
  sudo apt-get install sysstat
  ```
### Step 3: Set up Federated Server
- Create a Python virtual environment and install dependencies.
  ```bash
  python3 -m venv project
  source project/bin/activate
  pip install -e .
  pip install -r requirements.txt
  sudo apt-get install sysstat
  ```
### Step 4: Set up one Raspberry PI as a Web server
- Enable HTTP, TCP, and UDP ports.
### Step 5: Set up C&C Server as given in [Mirai](https://github.com/jgamblin/Mirai-Source-Code/) or [Bashlite](https://github.com/hammerzeit/BASHLITE/).
### Step 6: Connect one vulnerable Raspberry Pi client device, the Federated Server, and the C&C server and Web server with the same access point, and note down the IP address of all devices.
### Step 7: Start the packet_logger_append.py script to capture traffic in the Raspberry PI client device.
### Step 8: Generate normal network traffic by launching streaming applications or other commonly used internet-connected apps available on the Raspberry Pi devices.
### Step 9: Start scanning and injecting binaries from the C&C server as given in the respective malware codebase readme, and then launch a DDOS attack on a web server as mentioned in the readme. 
### Step 10: Stop all malware injection and label malicious based on the C&C server and Web server's IP address, and label benign to the rest of the samples in features.csv, which is captured by packet_logger_append.py
### Step 11: Split the labeled data and store it in all Raspberry Pi devices and the Federated server with the name data.csv in the home directory.
### Step 12: Connect all Raspberry Pi client devices, the Federated Server, the C&C server, and the Web server with the same access point.
### Step 6: To generate normal network traffic, launch streaming applications or other commonly used internet-connected apps available on the Raspberry Pi devices.
### Step 7: Start packet_logger.py script to capture traffic (this will create features.csv in the same directory where it is executing).
### Step 8: Start the FL server with 
```bash
flower-superlink --insecure
```
### Step 7: Start the Federated client on all Raspberry PI devices
``` bash
flower-supernode --insecure --superlink="<Federated Server IP>:9092" \
                 --node-config="dataset-path='path/to/features.csv', labeled-data='path/to/test.csv'"
```
Note : Add Federated Server IP, path to features.csv, which will be generated by packet_logger.py, and path to data.csv to labeled-data 
### Step 8: Start scanning and injecting binaries from the C&C server as given in the respective malware codebase readme. 
### Step 9: In the Federated server, go to fedmaldetect/embedded-devices and run
``` bash
  flwr run . embedded-federation
```
### Step 10: At the end of all communication rounds, test results will be available at each client's Flower terminal, and Memory/CPU utilization will be available in client_usage.log and server_usage.log (in server)
