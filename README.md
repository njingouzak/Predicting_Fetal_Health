# Predicting Fetal Health: A Comprehensive Classification Study
  
## Overview
In fetal health classification, our objective is to predict the well-being of unborn infants—an endeavor of profound importance that directly impacts expectant mothers and one of the most vulnerable populations. This study is based on a comprehensive fetal health classification dataset comprising 2,126 records derived from Cardiotocogram (CTG) examinations. Each record was labeled by three expert obstetricians into one of three categories: Normal, Suspect, or Pathological, providing a clinically reliable foundation for analysis. To achieve this objective, we employ three machine learning classifiers: logistic regression, random forest, and decision tree. Each model brings distinct analytical strengths, enabling a robust comparative assessment of their ability to identify patterns associated with fetal health outcomes. Through systematic training and evaluation, these models learn to distinguish between healthy conditions and potential signs of fetal distress.The resulting models support healthcare professionals in making timely, data-driven decisions that promote safer pregnancies and improved neonatal outcomes. Ultimately, this study highlights the transformative role of data-driven methods in healthcare, where analytical insights contribute to protecting and improving the health of future generations.  
  
## Project Goals
### Overall Goal
To develop and evaluate machine learning models that accurately classify fetal health status using Cardiotocogram (CTG) data, with the aim of supporting early detection of fetal distress and informed clinical decision-making.

### Specific Goals
- To explore and understand the fetal health dataset by Analyzing the distribution of fetal health classes (Normal, Suspect, Pathological)  
- To preprocess and prepare the data for modeling: handle missing values, address duplicates  
- To develop multiple classification models: logistic regression, decision tree, and random forest classifiers  
- To compare model performance: Assess models using accuracy score  
  
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
  
## Tools
Python
Pandas
Numpy
Skit-Learn
Jupyter Notebook






