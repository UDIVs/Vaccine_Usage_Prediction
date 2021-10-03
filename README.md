# Vaccine_Usage_Prediction

## Background:
We all have been going through this pandemic and I am sure nobody is enjoying this. The sooner this ends the better. The one think which helps in mitigating this outbreak is “herd immunity”. Now herd immunity can be achieved by vaccinating a large portion of population, but the problem arises when people are not willing to get vaccinated. 
In 2009, the Centres for Disease Control and Prevention (CDC) conducted a survey. 

This phone survey asked people whether they had received H1N1 and seasonal flu vaccines, in conjunction with information they shared about their lives, opinions, and behaviours. We were provided a small chunk of this dataset and posed the question: Using the survey results can we make a model that predicts who will get either vaccine?

## Problem Statement:
To anticipate how likely, it is that people will take the H1N1 flu vaccine. We'll also compare the accuracy of different models and choose the best one.

## Program:
We have to setup some methods/steps that we will be following to get our desired results:
1.	Exploratory Data Analysis
2.	About the models
3.	Comparing the model results
4.	Selection of the best model

### Exploratory Data Analysis
In this step we will be loading and cleaning the data. Cleaning which will involve missing value treatment, dropping the unwanted variables, changing categorical variables to machine understandable format, dummy variable creation, graphical interpretation and splitting the data set into train and test.

**Loading Dataset:** The dataset consists of 26,707 rows and 34 columns.

Columns                   | Description
--------------------------|-------------------------------------------------------
unique_id                 | Unique identifier for each respondent
h1n1_worry                | Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried
h1n1_awareness            | Signifies the amount of knowledge or understanding the respondent has about h1n1 flu - (0,1,2) - 0=No knowledge, 1=little knowledge, 2=good knowledge
antiviral_medication      | Has the respondent taken antiviral vaccination - (0,1)
contact_avoidance         | Has avoided any close contact with people who have flu-like symptoms - (0,1)
bought_face_mask          | Has the respondent bought mask or not - (0,1)
wash_hands_frequently     | Washes hands frequently or uses hand sanitizer -(0,1)
avoid_large_gatherings    | Has the respondent reduced time spent at large gatherings - (0,1)
reduced_outside_home_cont | Has the respondent reduced contact with people outside their own house - (0,1)
avoid_touch_face          | Avoids touching nose, eyes, mouth - (0,1)
dr_recc_h1n1_vacc         | Doctor has recommended h1n1 vaccine - (0,1)
dr_recc_seasonal_vacc     | The doctor has recommended seasonalflu vaccine -(0,1)
chronic_medic_condition   | Has any chronic medical condition - (0,1) 
cont_child_undr_6_mnth    | Has regular contact with child the age of 6 months -(0,1)
is_health_worker          | Is respondent a health worker - (0,1)
has_health_insur          | Does respondent have health insurance - (0,1)
is_h1n1_vacc_effective    | Does respondent think that the h1n1 vaccine is effective - (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective)
is_h1n1_risky             | What respondents think about the risk of getting illwith h1n1 in the absence of the vaccine- (1,2,3,4,5)-(1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky)
sick_from_h1n1_vacc       | Does respondent worry about getting sick by taking the h1n1 vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried,3=Doesn't know, 4=Respondent is somewhat worried,5Respondent is very worried) -
is_seas_vacc_effective    | Does respondent think that the seasonal vaccine is effective- (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective,5=Thinks it is highly effective)
is_seas_flu_risky         | What respondenst think about the risk of getting ill with seasonal flu in the absence of the vaccine- (1,2,3,4,5)- (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky)                                                                                                      
sick_from_seas_vacc       | Does respondent worry about getting sick by taking the seasonal flu vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried,3=Doesn't know, 4=Respondent is somewhat worried, 5Respondent is very worried)
age_bracket               | Age bracket of the respondent - (18 - 34 Years, 35 - 44 Years, 45 - 54 Years, 55 - 64 Years, 64+ Years)
qualification             | Qualification/education level of the respondent as per their response -(<12 Years, 12 Years, College Graduate, Some College) 
race                      | Respondent's race - (White, Black, Other or Multiple ,Hispanic)
sex                       | Respondent's sex - (Female, Male)
income_level              | Annual income of the respondent as per the 2008 poverty Census - (<= 75000−AbovePoverty,> 75000−AbovePoverty,>75000, Below Poverty)
marital_status            | Respondent's marital status - (Not Married, Married)
housing_status            | Respondent's housing status - (Own, Rent)
employment                | Respondent's employment status - (Not in Labor Force, Employed, Unemployed) 
census_msa                | Residence of the respondent with the MSA(metropolitan statistical area)(Non-MSA, MSANot Principle, CityMSA-Principle city) - (Yes, no)   
no_of_adults              | Number of adults in the respondent's house (0,1,2,3) - (Yes, no)
no_of_children            | Number of children in the respondent's house(0,1,2,3)- (Yes, No)
h1n1_vaccine              | Dependent variable)Did the respondent received the h1n1 vaccine or not(1,0) - (Yes, No)                                                               

#### Cleaning The Dataset:
- Missing Value Treatment: There were a lot of missing values in the dataset. Below is the list showing the columns and the count of null values in them.

| Columns                   | Count of Null Values |
|---------------------------|----------------------|
| unique_id                 | 0                    |
| h1n1_worry                | 92                   |
| h1n1_awareness            | 116                  |
| antiviral_medication      | 71                   |
| contact_avoidance         | 208                  |
| bought_face_mask          | 19                   |
| wash_hands_frequently     | 42                   |
| avoid_large_gatherings    | 87                   |
| reduced_outside_home_cont | 82                   |
| avoid_touch_face          | 128                  |
| dr_recc_h1n1_vacc         | 2160                 |
| dr_recc_seasonal_vacc     | 2160                 |
| chronic_medic_condition   | 971                  |
| cont_child_undr_6_mnths   | 820                  |
| is_health_worker          | 804                  |
| has_health_insur          | 12274                |
| is_h1n1_vacc_effective    | 391                  |
| is_h1n1_risky             | 388                  |
| sick_from_h1n1_vacc       | 395                  |
| is_seas_vacc_effective    | 462                  |
| is_seas_risky             | 514                  |
| sick_from_seas_vacc       | 537                  |
| age_bracket               | 0                    |
| qualification             | 1407                 |
| race                      | 0                    |
| sex                       | 0                    |
| income_level              | 4423                 |
| marital_status            | 1408                 |
| housing_status            | 2042                 |
| employment                | 1463                 |
| census_msa                | 0                    |
| no_of_adults              | 249                  |
| no_of_children            | 249                  |
| h1n1_vaccine              | 0                    |

- Outlier Treatment: There was no Outlier found in the dataset.
- Dummy Variable Creation: As we know, that machine learning algorithms cannot work on categorical variables. Therefore, we have created the dummy variables for the list of
  variables provided below.
  
| S.No. | Columns       | Rows  | Datatype |
|-------|---------------|-------|----------|
| 1     | age_bracket   | 26707 | object   |
| 2     | qualification | 26707 | object   |
| 3     | race          | 26707 | object   |
| 4     | sex           | 26707 | object   |
| 5     | income_level  | 26707 | object   |
| 6     | employment    | 26707 | object   |

- Discarding Redundant Variables: Given below are the variable which we have dropped which were either not relevant or had a relationship with other individual variables.
  - Unique_id
  - marital_status
  - housing_status
  - census_msa
  - no_of_children
  - no_of_adults

- Splitting the data into train and test: Split the data into train and test datasets and feed it into our model. Here we have split the data into 70% train and 30% test. The
  training and testing dataset consist of 18694 rows and 8013 rows correspondingly with 39 columns in both datasets. The train and test variables are given below.
  
| Colums                    |                         |                                 |                                    |
|---------------------------|-------------------------|---------------------------------|------------------------------------|
| h1n1_worry                | dr_recc_seasonal_vacc   | sick_from_seas_vacc             | race_OtherorMultiple               |
| h1n1_awareness            | chronic_medic_condition | age_bracket_35_44               | race_White                         |
| antiviral_medication      | cont_child_undr_6_mnths | age_bracket_45_54               | sex_Male                           |
| contact_avoidance         | is_health_worker        | age_bracket_55_64               | "income_level_greater_than$75,000" |
| bought_face_mask          | has_health_insur        | age_bracket_65+Years            | income_level_BelowPoverty          |
| wash_hands_frequently     | is_h1n1_vacc_effective  | qualification_less_than_12Years | income_level_Other                 |
| avoid_large_gatherings    | is_h1n1_risky           | qualification_CollegeGraduate   | employment_Notin_LaborForce        |
| reduced_outside_home_cont | sick_from_h1n1_vacc     | qualification_Other             | employment_Other                   |
| avoid_touch_face          | is_seas_vacc_effective  | qualification_SomeCollege       | employment_Unemployed              |
| dr_recc_h1n1_vacc         | is_seas_risky           | race_Hispanic                   |                                    |

- Graphical analysis: In this we analyse the given dataset to determine some other factors/situations using graphical visualizations.

![image](https://user-images.githubusercontent.com/89060175/135742445-46e76f63-909d-4368-904b-e6c4464ae29f.png)

Here we have a correlation heatmap of all the possible variables which shows the multicollinearity of the variables. As we can see apart from “employment_other” and “qualification_other” variable correlation no other variables are having high correlation. 

![image](https://user-images.githubusercontent.com/89060175/135742457-cb9f350a-009f-4733-812b-e329194b9ee1.png)

The graph above depicts the percentage of people who have received the vaccine, have had a seasonal vaccine advised by a doctor, are aware of H1N1, have had an H1N1 vaccine recommended by a doctor, and have avoided contact. To begin with, we can state that fewer people have been vaccinated. There must be a cause for this occurrence. As can be seen, more people have limited (1) or good (2) knowledge about H1N1, which is wonderful, but why aren't more people vaccinated?

Doctors are also not suggesting seasonal flu and H1N1 vaccines to many people; the reason for this may be seen in the most recent distribution, since most people are avoiding contact. This could aid in the prevention of disease spread. 

![image](https://user-images.githubusercontent.com/89060175/135742469-53e050aa-70e3-4815-850b-6c8e7b46f0a7.png)

The bar shows the h1n1 worry i.e., how many people are worried about the flu. Worry about the h1n1 flu (0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried. The graph shows the count of people who have taken the vaccine under which how many people are worried about the flu.

![image](https://user-images.githubusercontent.com/89060175/135742479-62b5983f-7a7c-4f1f-b08b-795a31a6b0c2.png)

From the above graph we can conclude that less health workers have taken the flu vaccine also very few non-health workers have taken the flu vaccine.

![image](https://user-images.githubusercontent.com/89060175/135742490-cb44e194-fafc-471d-90a7-d153f573f07e.png)

The bar graph depicts whether respondents are concerned about becoming ill as a result of receiving the seasonal flu vaccine - (1,2,3,4,5)- (1=respondent not concerned at all, 2=respondent is not very concerned, 3=doesn't know, 4=respondent is slightly concerned, 5=respondent is very concerned). It can be deduced that those who are concerned about becoming ill as a result of the vaccine are more likely to refuse the vaccination.

- Scaling Techniques: 
  - Normalization: Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max
    scaling.
    
    ![image](https://user-images.githubusercontent.com/89060175/135742539-3d570747-f7b4-4aec-9f4f-e137d52a9541.png)
    
  - Standardization: Standardization is another scaling technique where the values are centred around the mean with a unit standard deviation. This means that the mean of the
    attribute becomes zero and the resultant distribution has a unit standard deviation.
    
    ![image](https://user-images.githubusercontent.com/89060175/135742574-232ffb63-2879-4be7-9d85-964556f11871.png)

We have used scaling on train and test sets in only KNN because scaling techniques are irrelevant in Tree-based Algorithms as they are fairly insensitive to the scale of the features. A decision tree is only splitting a node based on a single feature. The decision tree splits a node on a feature that increases the homogeneity of the node. This split on a feature is not influenced by other features. So, there is virtually no effect of the remaining features on the split. This is what makes them invariant to the scale of the features.

### About the Models

1.	XGBoost: XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

2.	Adaboost: The AdaBoost algorithm, short for Adaptive Boosting, is a Boosting approach used in Machine Learning as an Ensemble Method. The weights are re-allocated to each
    instance, with higher weights applied to improperly identified instances. This is termed Adaptive Boosting. In supervised learning, boost is used to reduce bias and
    variation. It is based on the notion of successive learning. 


3.	KNN: The KNN algorithm believes that objects that are similar are close together. To put it another way, related items are close together. The KNN algorithm relies on this   
    assumption being correct in order for it to work. KNN calculates the distance between points on a graph to encapsulate the idea of similarity (also known as distance,
    proximity, or closeness).
    
    In K-nearest neighbour we have used both standardised dataset and normalised dataset and have then compared which one is working more efficiently.
    
    ![image](https://user-images.githubusercontent.com/89060175/135742639-740e5ee7-1b00-4bb8-a7bc-58bf0ae6f886.png)
    
    In above graph we have used standard scaler, and here we can see that both train and test accuracy is somewhat close at neighbours=19.
    
    ![image](https://user-images.githubusercontent.com/89060175/135742652-5702d57c-489a-4b2f-8d46-923e70005dd9.png)
    
    Whereas here we have used MinMax scaler and the accuracy between both train and test is close at neighbours= 25.
    
4.	RandomForest: Random Forest is made up of a huge number of independent decision trees that work together as an ensemble, as the name suggests. Each tree in the random
    forest produces a class prediction, and the class with the most votes become the prediction of our model.

5.	Decision Tree: It's a versatile tool that can be used in a variety of situations. Both classification and regression problems can be solved with decision trees. The name
    implies that it uses a tree-like flowchart to display the predictions that result from a sequence of feature-based splits. It begins with a root node and finishes with a
    leaf decision.

### Comparing the model results:
Below shown table gives the train and test accuracy of different models applied on the dataset.

| Model        | Train_Score | Test_Score |
|--------------|-------------|------------|
| Xgboost      | 0.924       | 0.85       |
| Adaboost     | 0.848       | 0.85       |
| KNN          | 0.841       | 0.838      |
| DecisionTree | 0.855       | 0.847      |
| RandomForest | 0.897       | 0.854      |

### Selection Of Model:
From the above analysis we can say that Xgboost and Random forest model is overfit i.e. train score is greater than test score. Only Adaboost is having a low bias/variance trade-off, therefore using Adaboost gives us the optimum result.








                                                                                                                        
