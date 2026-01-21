# Predicting Fetal Health: A Comprehensive Classification Study
---  
## Overview
In fetal health classification, our objective is to predict the well-being of unborn infants—an endeavor of profound importance that directly impacts expectant mothers and one of the most vulnerable populations. This study is based on a comprehensive fetal health classification dataset comprising 2,126 records derived from Cardiotocogram (CTG) examinations. Each record was labeled by three expert obstetricians into one of three categories: Normal, Suspect, or Pathological, providing a clinically reliable foundation for analysis. To achieve this objective, we employ three machine learning classifiers: logistic regression, random forest, and decision tree. Each model brings distinct analytical strengths, enabling a robust comparative assessment of their ability to identify patterns associated with fetal health outcomes. Through systematic training and evaluation, these models learn to distinguish between healthy conditions and potential signs of fetal distress.The resulting models support healthcare professionals in making timely, data-driven decisions that promote safer pregnancies and improved neonatal outcomes. Ultimately, this study highlights the transformative role of data-driven methods in healthcare, where analytical insights contribute to protecting and improving the health of future generations.  

 ---
## Project Goals
### Overall Goal
To develop and evaluate machine learning models that accurately classify fetal health status using Cardiotocogram (CTG) data, with the aim of supporting early detection of fetal distress and informed clinical decision-making.

### Specific Goals
- To explore and understand the fetal health dataset by Analyzing the distribution of fetal health classes (Normal, Suspect, Pathological)  
- To preprocess and prepare the data for modeling: handle missing values, address duplicates  
- To develop multiple classification models: logistic regression, decision tree, and random forest classifiers  
- To compare model performance: Assess models using accuracy score
- To Assess Fetal Health Predictions

---
## Dataset
This dataset comprises a comprehensive collection of physiological parameters recorded during pregnancy and labor to assess the well-being of the fetus. It includes the following features:  
  
**- baseline value:** This typically refers to the baseline fetal heart rate (FHR) measured in beats per minute (bpm) during a specific period of time before any contractions or other events occur.  
  
**- accelerations:** These are temporary increases in the fetal heart rate (FHR) above the baseline, typically associated with fetal movement or other non-pathological stimuli. Accelerations are considered a reassuring sign of fetal well-being.  
  
**- fetal_movement:** This may refer to the presence or absence of fetal movement recorded during monitoring. Fetal movement can provide insight into fetal well-being and neurologic development.  
  
**- uterine_contractions:** These are contractions of the uterus that occur during labor. Monitoring the frequency, duration, and intensity of contractions is important for assessing labor progress and fetal well-being.  
  
**- light_decelerations:** These are temporary decreases in the fetal heart rate (FHR) below the baseline, typically of short duration and amplitude. Light decelerations are often associated with uterine contractions and are considered a normal response to fetal head compression during labor.  
  
**- severe_decelerations:** These are pronounced and prolonged decreases in the fetal heart rate (FHR) below the baseline, often indicating compromised fetal oxygenation and potential fetal distress.  
  
**- prolongued_decelerations:** These are decelerations in the fetal heart rate (FHR) that last longer than a certain duration, typically defined as lasting more than two minutes. Prolonged decelerations can be indicative of fetal hypoxia and may require medical intervention.  
  
**- abnormal_short_term_variability:** This refers to irregular fluctuations in the fetal heart rate (FHR) that occur over short periods of time. Abnormal short-term variability can be a sign of fetal distress.  

**- mean_value_of_short_term_variability:** This represents the average amplitude of short-term variability in the fetal heart rate (FHR), which is an important parameter for assessing fetal well-being.
  
**- percentage_of_time_with_abnormal_long_term_variability:** This indicates the proportion of time during monitoring that the long-term variability in the fetal heart rate (FHR) deviates from the normal range. Abnormal long-term variability can be associated with fetal distress.  
  
**- mean_value_of_long_term_variability:** This represents the average amplitude of long-term variability in the fetal heart rate (FHR), which is another important parameter for assessing fetal well-being.  
  
**- histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency:** These parameters are derived from the fetal heart rate (FHR) histogram, which is a graphical representation of the distribution of FHR values over a period of time. Each of these parameters provides information about the shape, central tendency, and variability of the FHR histogram, which can be used to assess fetal well-being.  
  
**- fetal_health:** This is the target variable that indicates the fetal health outcome, typically categorized into different classes (e.g., normal, suspect, pathological) based on various physiological parameters and clinical assessments.  

---
## Tools
- Python  
- Pandas  
- Numpy  
- Scikit-Learn  
- Jupyter Notebook  

---
## Approach
  
### Module 1
### Task 1: Data Dive for Exploring Fetal Health Insights
Access the fetal health data for cleaning and analysis to ensure we have accurate input for informed decision-making in prenatal care.    
    
### Task 2: Managing Duplicates in Fetal Health Data
Ensure data integrity by identifying duplicate records that could distort analysis and lead to inaccurate insights.  
    
### Task 3: Enhancing Data Integrity
Clean the dataset by removing duplicate records to ensure that every observation is unique, which is crucial for reliable analysis and model training. After identifying duplicate records, we now need to remove them from the dataset. This task ensures that each fetal health observation is counted only once.  
  
### Task 4: Managing Missing Values in Fetal Health Dataset
Before using the data for predictive modeling, it's essential to ensure that there are no gaps or missing values that could compromise the analysis. From a business perspective, ensuring data completeness leads to more reliable decision-making.     
    
### Task 5: Class Distribution Analysis
For healthcare professionals to trust model predictions, it’s crucial that the data represents all health conditions fairly. In this project, we will analyze the distribution of fetal health categories to ensure balanced representation. Understanding the frequency of each class in the fetal health data helps to identify potential imbalances, guiding decisions on whether to adjust or balance the dataset prior to modeling.     
    
### Module 2
### Task 1: Split Data into Features and Target
For predictive modeling to be effective, it's critical that the model is trained on the correct data. Separating the input features from the target variable ensures that the model learns the relationship between predictors and outcomes without any data leakage. In this case, the objective is to split the fetal health dataset into two parts: one that contains all the input features and another that contains the target variable, which is the fetal health status. This division is key to building robust and reliable machine learning models.   
    
### Task 2: Split Data into Training and Testing Sets
To develop a robust predictive model that performs well in real-world scenarios, it is essential to evaluate the model on data it has never seen before. By splitting the dataset into training data (used to build the model) and testing data (used to evaluate its performance), we ensure unbiased validation of the model’s predictive capabilities. This separation helps to detect overfitting and provides a realistic estimate of how the model will perform on new data.   
    
### Task 3: Train a Logistic Regression Model
To provide healthcare professionals with actionable insights, it's essential to have a model that can accurately classify fetal health. The goal is to quickly identify potential risks by building a predictive model. Here, we'll build and train a logistic regression model—a simple yet powerful classification algorithm—to serve as a baseline for further comparisons with more complex models.     
    
### Task 4: Make Predictions with the Trained Model
Once the logistic regression model is trained, it’s critical to evaluate how well it performs on new, unseen data. This step is key to ensuring that the model can reliably predict fetal health in real-world scenarios. By generating predictions on the test dataset, healthcare professionals can assess the model's potential to identify cases that require further medical attention.     
    
### Task 5: Calculate Model Accuracy
For healthcare decision-making, it’s crucial to have confidence in the model’s predictions. By calculating the accuracy, we validate the model’s performance and ensure that it reliably identifies fetal health outcomes. High accuracy means that the model's predictions closely match the actual patient data, which is vital for making informed clinical decisions.   
    
### Task 6: Initialize a Random Forest Classifier
In healthcare, making accurate predictions is vital for early detection and intervention. To capture more complex patterns in fetal health data and improve predictive accuracy, we need an advanced model. This task involves setting up a Random Forest classifier—a robust ensemble learning method—to leverage multiple decision trees and generate more reliable predictions. This approach not only enhances the model's ability to generalize from the data but also provides deeper insights into patient risk profiles.   
    
### Task 7: Train the Random Forest Classifier
To support clinical decision-making, the model must be trained to accurately classify fetal health based on complex data patterns. In this task, we will train the previously initialized Random Forest classifier on the training dataset. This process enables the model to learn from historical data and capture intricate relationships, which are essential for making robust predictions in real-world healthcare settings.     
    
### Task 8: Make Predictions with the Random Forest Model
To ensure our advanced predictive system is reliable in clinical settings, it's crucial to evaluate how well our Random Forest model can classify fetal health outcomes on new, unseen cases. Accurately predicting these outcomes is vital for early risk detection and timely medical intervention. In this task, we'll use the trained Random Forest model to generate predictions on the testing dataset.     
  
### Task 9: Calculate Accuracy of the Random Forest Model
In clinical settings, having a reliable and accurate predictive model is essential for making informed decisions. To ensure that our advanced Random Forest model can be trusted for early risk detection, we need to evaluate its performance by calculating its accuracy. This metric tells us the proportion of correct predictions, providing a quick measure of the model's overall reliability.     
    
### Task 10: Initialize a Decision Tree Classifier
In a clinical setting, it is essential to provide not only accurate but also interpretable predictions. Clinicians often prefer models that are easy to understand and explain. A Decision Tree classifier offers a transparent decision-making process, making it ideal for explaining how predictions are made in fetal health classification. By initializing a Decision Tree with controlled complexity, we can offer clear insights into the factors influencing each prediction.    
    
### Task 11: Train the Decision Tree Classifier
In clinical settings, it is vital to not only predict fetal health outcomes accurately but also to provide a transparent, interpretable decision-making process. A Decision Tree model is particularly valuable for this purpose because it allows clinicians to see the exact decision rules leading to each prediction. The business goal is to build a model that learns simple, actionable rules from historical patient data to classify fetal health reliably.    
    
### Task 12: Make Predictions with the Decision Tree Model
In a clinical environment, it’s essential that predictions are both accurate and understandable. A Decision Tree model, with its transparent decision-making process, offers clinicians clear insights into how predictions are made. The business goal here is to generate predictions regarding fetal health that are easily interpretable, thereby increasing trust and facilitating informed decision-making.     
    
### Task 13: Calculate Accuracy of the Decision Tree Model
In clinical applications, even simpler, interpretable models must be rigorously evaluated to ensure that they provide reliable predictions. The goal here is to verify that our straightforward Decision Tree model delivers accurate classifications of fetal health outcomes. By measuring its accuracy, we gain a clear metric that shows how well the model's predictions match the actual clinical data, thus supporting its use in decision-making.     
    
### Module 3
### Task 1: Model Performance Unveiled: Assessing Fetal Health Predictions
For effective clinical decision-making, it is essential that the performance of predictive models is both transparent and easily interpretable. In this task, the goal is to compare the actual fetal health outcomes with the predictions made by the advanced Random Forest model. By creating a side-by-side comparison table, stakeholders can quickly see where the model performs well and identify areas that may require further tuning. This visualization supports confidence in the model's ability to aid in early risk detection.     

---      
## Key Outcomes
After preprocessing and preparing the fetal health data, we developed three classification models using different machine learning algorithms. While all models demonstrated strong performance, the random forest classifier achieved the highest accuracy, with a score of **0.921**. Consequently, this model was selected to generate and assess predictions of fetal health outcomes.  

--- 
## Conclusion
In obstetrics, the assessment of fetal health remains a critical yet challenging task. Traditional approaches based on Electronic Fetal Monitoring (EFM) continue to face limitations, particularly regarding their impact on neonatal outcomes and the reduction of emergency cesarean deliveries. To address these challenges, this study proposes an advanced machine learning–based approach for fetal health prediction, utilizing an ensemble of classification algorithms, including Logistic Regression, Decision Tree, and Random Forest. Among the evaluated models, the Random Forest classifier demonstrated superior performance, achieving the highest accuracy and consistently outperforming the other methods. This result highlights its effectiveness in accurately identifying and classifying varying fetal health conditions. Overall, the findings underscore the potential of machine learning techniques to enhance fetal health monitoring by providing a more objective, reliable, and comprehensive framework for assessing fetal well-being, with significant implications for prenatal care and clinical decision-making.  

 ---
 ## Acknowledgements
 I would like to express my sincere gratitude to the leadership of the **HiCounselor platform** for providing the resources and guidance that supported and facilitated the successful completion of this project.  

---


