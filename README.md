# Image-Classification
# Introductions
In recent years, the rapid urbanization and increase in vehicle ownership have intensified the demand for efficient and effective parking management systems. 
Traditional methods of parking management, which often rely on manual supervision and static signage, are increasingly proving to be inadequate in coping with the growing complexity and scale of urban parking needs.
Consequently, there is a pressing need for intelligent and automated systems that can enhance the efficiency of parking space utilization and reduce the time spent by drivers searching for available spots.
This research aims to address this need by leveraging advanced machine learning techniques, specifically Support Vector Machines (SVM) and Convolutional Neural Networks (CNN), for the detection and management of parking spaces.

# Data Source
The data used in this study  is a public dataset from https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter. 
The main problem involving the classification of parking space is the lack of consistent and reliabel datasets.
This dataset is designed for the task of detecting and counting empty and occupied parking spots in a parking area.
It includes images, a mask image, a video, and a utility file (util.py) that provides functions to process these images and video frames.

# Step Of Analysis
The analysis steps used in this research are as follows:
1.	Import the parking lot images labeled as ‘empty’ and ‘not empty’.
2.	Flatten the images to a consistent size for uniformity.
3.	Resize images to a standardized dimension to ensure compatibility with the models.
4.	Split the dataset into training and testing sets to evaluate model performance.
5.	Visualize sample data from the training set to understand the distribution and characteristics of the images
6.	Prepare for training the models by setting up the necessary configurations.
7.	SVC Path :
1.	Train the SVC model using the training data.
2.	Optimize the hyperparameters of the SVC model for better performance.
3.	Evaluate the SVC model using testing data and calculate performance metrics.
4.	Select the best SVC model based on evaluation metrics.
8.	CNN Path :
1.	Train the CNN model using the training data.
2.	Optimize the hyperparameters of the CNN model for enhanced accuracy.
3.	Evaluate the CNN model using testing data and compute performance metrics.
4.	Choose the best CNN model based on evaluation results.
9.	CNN – SVM Path :
1.	Train the CNN model using the training data.
2.	Optimize the hyperparameters of the CNN model for enhanced accuracy.
3.	Evaluate the CNN model using testing data and compute performance metrics.
4.	Choose the best CNN model based on evaluation results.
10.	Compare the best models from SVM, CNN, and CNN – SVM model.
11.	Select the superior model based on comparative analysis.
12.	Import the selected model into the utils.py script for deployment.
13.	Utilize the utils.py script to perform parking lot detection with the chosen model.
14.	Conclude the process with a functional parking space detection system.

# Flow Chart Step Of Analysis
![image](https://github.com/Nexus-Consultant/Image-Classification/assets/172244503/2992b524-2efa-443c-bdcd-9060b64871f3)

# Model Deafult
## Support Vector Machine
## Neural Network
## Neural Network-Support Vector Machine
# Best Model
According to our modeling result, we choose the SVM model as the best model as it’s has the shortest running time and performs very well for the given data.
The SVM model that we’re choose with it’s hyperparameter will be shown below.
![image](https://github.com/Nexus-Consultant/Image-Classification/assets/172244503/afa5177f-60aa-4679-85d3-09766695c1a5)

# Conclusion
•	Effectiveness of SVM and CNN: Both SVC and CNN models demonstrated high accuracy in detecting parking space occupancy, with SVC achieving an accuracy of 0.996 and CNN achieving 0.994 after optimization.The hybrid CNN-SVM model also showed comparable performance with an accuracy of 0.994.
•	Efficiency Considerations: Despite the similar accuracy levels among the models, the SVC model was identified as the most efficient due to its shorter runtime. This makes SVC a more practical choice for real-time applications in parking management systems.
•	Model Deployment: The deployment of the SVC model in a real-time image processing system successfully demonstrated its practicality for parking space detection. The system efficiently processed video frames to detect occupancy status and handled dynamic changes in parking lot conditions.
•	Data Characteristics: The balanced nature of the dataset and the distinct visual differences between empty and occupied parking spaces contributed to the high performance of all models. This highlights the importance of dataset quality and labeling in machine learning applications.

# Reference
[1]  Akay, M. F. (2009). Support vector machines combined with feature selection for breast cancer diagnosis. Expert Systems with Applications, 36(2), 3240-3247.
[2]	Amato, G., Carrara, F., Falchi, F., Gennaro, C., Meghini, C., & Vairo, C. (2016). Deep learning for decentralized parking lot occupancy detection. Expert Systems with Applications, 72, 327-334.
[3]  Apostolopoulos, D., Markou, I., & Panagiotakis, S. (2018). DeepSpot: Parking Space Detection using Deep Learning. Sensors, 18(11), 3787.
[4]   Burges, C. J. C. (1998). A tutorial on support vector machines for pattern recognition. Data Mining and Knowledge Discovery, 2(2), 121-167.
[5]   Chang, C. C., Lin, C. J. (2002). LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1–27:27.
[6]	Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
[7]	Funck, W., Mögelmose, A., & Trivedi, M. M. (2004). Real-time multicamera vehicle detection for traffic surveillance and active safety applications. IEEE Transactions on Intelligent Transportation Systems, 15(3), 118-129
[8]   Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
[9]	Huang, X., & Wang, J. (2010). A method of parking space detection based on image processing. IEEE Transactions on Intelligent Transportation Systems, 11(2), 359-367.
[10]	Ichihashi, H., Notsu, A., Honda, K., Katada, T., & Fujiyoshi, M. (2009). Vacant parking space detector for outdoor parking lot by using surveillance camera and FCM classifier. Fuzzy Sets and Systems, 160(1), 89-103. [1]	G. O. Young, “Synthetic structure of industrial plastics,” in Plastics, 2nd ed., vol. 3, J. Peters, Ed. New York: McGraw-Hill, 1964, pp. 15–64.
[11]  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 1097-1105.
[12]  LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
[13]	LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
[14]  Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), 807-814.
[15]  Noble, W. S. (2006). What is a support vector machine? Nature Biotechnology, 24(12), 1565-1567.
[16]	Paidi, R., Fleyeh, H., Håkansson, J., & Nyberg, R. G. (2018). Smart parking sensors, technologies and applications for open parking lots: A review. IET Intelligent Transport Systems, 12(8), 735-741.
[17]  Rosasco, L., De Vito, E., Caponnetto, A., Piana, M., & Verri, A. (2004). Are loss functions all the same?. Neural Computation, 16(5), 1063-1076.
[18] Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
[19]	Tang, J., Bi, K., & He, D. (2006). Parking space detection method based on image processing. In 2006 International Conference on Mechatronics and Automation (pp. 3094-3099). IEEE. 
[20]	Xu, Z., Wang, L., & Xu, W. (2017). A hybrid approach to parking space detection using deep learning and SVM. In 2017 IEEE International Conference on Computer Vision Workshops (ICCVW) (pp. 34-40). IEEE. 
