# VLF_VRF
Vessel Location/Route Forecasting


# Description 

This repository provides an improved version of the core parts of the code that implements the VLF and VRF methods using Tensorflow 2; note that the original VLF method was implemented by using Tensorflow 1. The code for training the models is divided into two parts: a) "VLF_VRF_train_config.py" which includes the algorithms' parameters, and b) "VLF_VRF_train_main.py" which is the main code and builds the model through the function "build_train_simple_model" by using four classes: 
- “Dataset”: Loads a csv file that includes the available dataset and performs data cleansing.
- “DatasetProcessed”: Performs trajectory segmentation, splits the available dataset into 3 subsets (training, validation, testing) and shuffles the resulted sub-trajectories.
- “DatasetTransformed”: Transforms trajectories into a supervised problem of 3D arrays of training, validation and testing sets. Also, it sets the burned points (necessary points for initializing the LSTM states) and the minute labels. Furthermore, it applies zero padding to the trajectories and normalizes the data.
- “Predictions”: Produces final predictions

In order to train a model you should run the "VLF_VRF_train_main.py" by setting the relative parameters in the "VLF_VRF_train_config.py". Note that the VLF model can be trained when in the "VLF_VRF_train_config.py" the parameter "cfg_look_ahead_points" is equal to 1. A subset from the Brest dataset (with anonymised mmsi) is given as an example.

Also, this repository provides the code for online prediction by using the trained VLF model. This part of the code is based on Apache Kafka and Zookeeper, which can be installed according to: https://kafka.apache.org/quickstart
In order to make online predictions you should train a model and then run the "VLF_VRF_pred_online_main.py" by setting the parameters in the "VLF_VRF_pred_online_config.py". Note that the configuration parameters for the training and the prediction phases should be the same. A trained model is given as an example. 


# References

If you use the code for training models, please cite the corresponding papers:
E. Chondrodima, N. Pelekis, A. Pikrakis and Y. Theodoridis, "An Efficient LSTM Neural Network-Based Framework for Vessel Location Forecasting," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2023.3247993.
E. Chondrodima, P. Mandalis, N. Pelekis and Y. Theodoridis, "Machine Learning Models for Vessel Route Forecasting: An Experimental Comparison," 2022 23rd IEEE International Conference on Mobile Data Management (MDM), Paphos, Cyprus, 2022, pp. 262-269, doi: 10.1109/MDM55031.2022.00056.
P. Mandalis, E. Chondrodima, Y. Kontoulis, N. Pelekis and Y. Theodoridis, "Towards a Unified Vessel Traffic Flow Forecasting Framework," EDBT/ICDT 2023 Joint Conference, Ioannina, Greece, 2023

If you use the code for online prediction, please cite the corresponding paper:
P.Tampakis, E.Chondrodima, A.Pikrakis, Y.Theodoridis, K.Pristouris, H.Nakos, E.Petra, T.Dalamagas, A.Kandiros, G.Markakis, I.Maina, S.Kavadas, "Sea Area Monitoring and Analysis of Fishing Vessels Activity: The i4sea Big Data Platform," 2020 21st IEEE International Conference on Mobile Data Management (MDM), Versailles, France, 2020, pp. 275-280, doi: 10.1109/MDM48529.2020.00063.


