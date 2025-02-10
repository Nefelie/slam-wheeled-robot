# SLAM and Perception System Development for an Intelligent Wheeled Robot
This project involves the development of software for an intelligent mobile robot using the ZeroROS robot middleware and the Webots simulation environment. The primary focus was on building a robust perception system and implementing simultaneous localization and mapping (SLAM) algorithms.

Please see reports below of individual contribution. 
- [Motion Planning and Control](https://drive.google.com/file/d/19nkKd4coG3xoGSrkOfAIy0G_KUmcwSe0/view?usp=drive_link)
- [Probabilistic Localisation](https://drive.google.com/file/d/1DuddWuD80t55BdlHlDozhPjqzn_1oSd_/view?usp=drive_link)   
- [Cognition/SLAM](https://drive.google.com/file/d/1OlaRMtMkpjJAo23eWf6qocxDeh-sPvJa/view?usp=drive_link)   


![maxresdefault](https://github.com/user-attachments/assets/d639fc2d-f2f8-428a-b47d-c76bc0a8a63c)

Key accomplishments include:
- State-Space Control: Implemented a state-space controller to ensure precise and stable robot motion.
- Probabilistic Localization: Developed an Extended Kalman Filter (EKF) and Particle Filter for accurate probabilistic localization.
- LiDAR Data Interpretation: Processed live LiDAR sensor data using a Gaussian Process Classifier and Regressor to detect and classify walls and corners in the robot's environment.
- Graph SLAM: Created a Graph SLAM algorithm to construct a detailed and accurate map of the surroundings while simultaneously localizing the robot within it.
