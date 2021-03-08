# OPTIMIZATION OF SARCASM DETECTION IN TWEETS USING SUPERVISED MACHINE LEARNING MODEL

**abstract**: 
The Sarcasm is used to mock or convey contempt through a sentence or while speaking. People apply positive words to reveal gloomy feelings. The presence of sarcasm poses a challenge in sentiment analysis and causes the misclassification of people’s sentiment. Hence, it leads to reduce the accuracy of sentiment analysis. Sentiment analysis is an approach to classify and elicit opinions towards any particular interest like products, events, services, topics and so on.  The implementation of a robust and efficient system to detect sarcasm can be one way to improve accuracy for sentiment analysis. To recognize sarcasm, we applied a set of machine learning classification algorithms along with a variety of features to identify the best classifier model with significant features which lead to recognize the sarcasm in tweets to get better performance of sentiment analysis. We suggest amendments, for instance, selection of the right set of features which lead to get better accuracy, which is presented in the result analysis part of this paper. Analysis results show that Decision Tree (91.84%) and Random Forest (91.90%) outperform the accuracy compared to Logistic Regression, Gaussian Naive Bayes and Support Vector Machine for the different features selection.

# Replication package Structure:
```
📁 Sarcasm_detection_package/
├─ 📁 Dataset/
|─ 📁 Features/
├─ 📁 Scripts/
├─ 📁 Results/
| 
─
```

# How to run:
  1. Download the dataset from [https://github.com/syful-is/Sarcasm_detection_package.git](https://github.com/syful-is/Sarcasm_detection_package.git)
  2. Extract the files. 
  3. Clone this repository into your userhome folder in the system and run the code
  ```https://github.com/syful-is/Sarcasm_detection_package.git```
  3. Open `Jupyter Notebook` or `Python Spyder`.
  4. Copy any code and Set your working directory using 
                
                ```
                import os
                
                #Please specify your dataset directory. 
                os.chdir("..../Dataset/")
                ```
                
     
  5. Install the dependencies running pre_setup.py in scripts folder
  6. Execute feature_engineering.py in scripts folder to extract features which are used to produce results.
  7. Finally, run sarcasm_classifiers.py in scripts folder to get results for individual classifier or algorithm NB: You may have to install required packages depending on the python environment you are using.
  
  
# Authors:
  1. [Arifur Rahman,  MSc student, Dept. of CSTE NSTU, Bangladesh](https://nstu.edu.bd/department/cste)
  2. [Syful Islam, Nara Institute of Science and Technology (NAIST), Nara, Japan.](https://syful-is.github.io/)
  3. [Ratnadip kuri](https://nstu.edu.bd/faculty-member/ratnadip-kuri-yky939)
  4. [Associate professor Md. Javed Hossain](https://nstu.edu.bd/faculty-member/md-javed-hossain-bdr991)
  5. [Professor Dr. Humayun Kabir](https://nstu.edu.bd/faculty-member/dr-humayun-kabir-95c631)
  

