# Personal Injury Prediction In Traffic Accidents - Montgomery County, MD

## Project Goals

* The goal is by analyzing the past ten years' traffic accident records in Montgomery county Maryland. Find the key drivers of personal injury in those traffic accidents. Make recommendations to the police department to better prediction on personal injury in traffic accidents so the casualty would get treated as early as possible.

## Project Description

* In this report, I will use 2012-01-01 to 2022-08-15 Montgomery County (MD) traffic violation data. Use the classification machine learning method to develop a model to predict personal injury in traffic accidents. It will help the police department has a better preparation of the medical resources for casualties in accidents.
* In the end, I will give out recommendations for the data collection system and the next step I would like to take.

## Initial Questions

1. Does personal injury in an accident related to the hour? or day of the week? month? year?
2. Does the personal injury in accident rate related to race and gender?
3. What about the relationship between personal injury and Violation Type? alcohol?
4. Does personal injury in the accident have a relationship with the agency location?
5. Is property damage and belts use have a relationship with personal injury?

## Data Dictionary

Variables are used in this analysis:

* Date Of Stop: Date of the traffic violation.
* Time Of Stop: Time of the traffic violation.
* SubAgency: Court code representing the district of assignment of the officer. 
* Belts: YES if seat belts were in use in accident cases.
* Personal Injury: Yes if traffic violation involved Personal Injury.
* Property Damage: Yes if traffic violation involved Property Damage.
* Alcohol: Yes if the traffic violation included an alcohol related suspension.
* Violation Type: Violation type. (Examples: Warning, Citation, SERO)
* Contributed To Accident: If the traffic violation was a contributing factor in an accident.
* Race: Race of the driver. (Example: Asian, Black, White, Other, etc.)
* Gender: Gender of the driver (F = Female, M = Male)

## Steps to Reproduce

1. Download the CSV file from https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q.
2. Import CSV file to jupyter notebook. Set up data period: 2012-01-01 to 2022-08-15,
3. Clone my repo (including the acquire_zillow.py and prepare_zillow.py) (confirm .gitignore is hiding your env.py file)
4. Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, scipy.stats, math.
5. You should be able to run final_report.

## Plan

## Wrangle

### Modules (acquire.py + prepare.py)

1. Download the CSV file from https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q. Data period: 2012-01-01 to 2022-08-15.
2. Add acquire function to acquire.py module
3. Write code to clean data in notebook
4. Write code to split data in notebook
6. Add clean and split functions (or more) to prepare.py file
9. Import into notebook and test functions.

### Acquire the Zillow data

To acquire the Montgomery county traffic violation data, I did the following two steps:
1. Download the CSV file from https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q.
2. Import CSV file to jupyter notebook.
* Data period: 2012-01-01 to 2022-08-15, if you want to download the data and run the notebook, be aware the date.

### Missing Values (report.ipynb)

No missing values.

### drop columns
'SeqID', 'Agency', 'Accident', 'Fatal', 'Commercial License','HAZMAT', 'Commercial Vehicle', 'Work Zone', 'State', 'VehicleType', 'Year', 'Make', 'Model', 'Color', 'Charge', 'Article', 'Driver State', 'DL State', 'Search Reason For Stop', 'Search Arrest Reason', 'Location','Search Conducted', 'Search Disposition', 'Search Outcome', 'Search Reason', 'Search Type','Geolocation','Latitude', 'Longitude','Description', 'Driver City', 'Arrest Type'

### drop rows
SubAgency S15 only has 3 rows and W15 only have 7 rows. drop those 10 rows.

### Convert data type
convert time format. set the time to index and create new columns for year, month, day of the week and hour.

convert the boolean value into int: Contributed to accident

### Set up a cut-off line to analyze the accident data
get all the data related to accident.

### Data Split (prepare.py (def function), report.ipynb (run function))

* train = 56%
* validate = 24%
* test = 20%

### Using your modules (report.ipynb)
once acquire.py and prepare.py are created and tested, import into final report notebook to be ready for use.

## Set the Data Context

There are 48,076 accidents related in Montgomery county, Maryland traffic violation records.

The data time range is from 2012-01-01 to 2022-08-15.

Due to COVID, traffic violations and accident numbers reduce a lot from 2020.

1. plot a displot for year accident counts, x is year, y is count
2. plot a displot for gender counts, x is gender, y is count
3. plot a displot for race counts, x is race, y is count

## Explore

1. Does personal injury in an accident related to the hour? or day of the week? month? year?
2. Does the personal injury in accident rate related to race and gender?
3. What about the relationship between personal injury and Violation Type? alcohol?
4. Does personal injury in the accident have a relationship with the agency location?
5. Is property damage and belts use have a relationship with personal injury?

## Exploring through visualizations (report.ipynb)

1. Does personal injury in an accident related to the hour? or day of the week? month? year?

Answer:
* Hour: Most personal injury accidents happen during the daytime, from 06:00 - 18:00.
* Day of the week: Weekends seem to have fewer personal injury accidents even weekends have more accidents number.
* Month: June and October have the highest personal injury rate in accidents just the same as accidents number.
* Year: 2020-2022 has the lowest rate because COVID started from the beginning of 2020 and accidents number reduced a lot due to COVID.
* run Chi2 statistic tests for day_of_week and personal injury, hour and personal injury.

2. Does the personal injury in accident rate related to race and gender?

Answer:
* Gender: male accident number is more than double of the female, but the female's personal injury rate is higher than males.
* Race: according to the race count and personal injury rate in the accident, white people should be the highest one.

3. What about the relationship between personal injury and Violation Type? alcohol?

Answer:
* Violation Type: Citation and warning are the highest two, it might be because citation has the highest accident counts.
* Alcohol: alcohol-related accident has more chance to cause personal injury

4. Does personal injury in the accident have a relationship with the agency location?

Answer:
* Different locations have different rates of personal injury, they are dependent. Germantown is the highest one.

* run Chi2 statistic tests for agency locaion and personal injury.

5. Is property damage and belts use have a relationship with personal injury?

Answer:
* it seems property damage and personal injury has no relationship.
* belts nummber seems hard to explain, maybe because some of the violatons don't require the belt check.

## Summary (report.ipynb)

Race, gender, alcohol, agency location and all the time features all have relationships with personal injury in traffic accidents. Also, violation type is related to our target variable personal injury. Since violation type is an outcome of traffic violations, personal injury is also an outcome. I prefer to not use one outcome to predict another outcome.

* The feature will be used for feature engineering will be:

    Alcohol, race, gender, subagency, hour, day of the week, month

## Feature Engineering

* select K best : race, day of week, hour, subagency
* RFE : month


# Modeling

## Select Evaluation Metric (Report.ipynb)

* Because personal injury in accidents is a boolean/yes or no value, I will use classification machine learning algorithms to build my models.
* Here I will use race, agency location, hour, week of day and month as my features. Then build four different models with the same features.
    1. KNN
    2. Decision tree
    3. Random forest
    4. Logistic regression

For the metric, I will use F1 score because I want to minimize all the false predictions. If there is personal injury in accidents but we don't send out medical assistance, it might cause the casualty couldn't get treatment in time. Or if there is no personal injury but we send medical assistance, it will cause a waste of resources.
* For calculating F1 score, I created a function in prepare.py

## Evaluate Baseline (Report.ipynb)

The baseline value I set for train and validate set is the mode of personal injury in accidents on the train set and validate set.

## Develop 4 Models (Report.ipynb)

1. KNN
2. Decision tree
3. Random forest
4. Logistic regression

## Evaluate on Train (Report.ipynb)

* Accuracy on train: 
    * baseline -- 77.10%
    1. KNN -- 81.61%
    2. Decision tree -- 80.82%
    3. Random forest -- 83.34%
    4. Logistic regression -- 77.09%
    
* F1 score on train:
    1. KNN -- 41.82%
    2. Decision tree -- 31.42%
    3. Random forest -- 43.46%

* Takeaway:
it seems logistic regression model doesn't perform well on imbalance data, there is no need to calculate the F1 score.

## Evaluate on Validate (Report.ipynb)

* Accuracy on validate:     
    * baseline -- 77.53%
    1. KNN -- 79.25%
    2. Decision tree -- 79.08%
    3. Random forest -- 81.13%


* F1 score on validate:
    1. KNN -- 30.73%
    2. Decision tree -- 24.06%
    3. Random forest -- 31.17%
    
* Random forest is the best model

## Evaluate Top Model on Test (Report.ipynb)

test result:
* Accuracy: 81.07%
* F1 score: 31.11%

## Expectation:
According to the test result, I expect the model will perform 81.07% accuracy in the future data if the data souce has no major change.

# Report (Final Notebook)

## Code commenting (Report.ipynb)

## Written Conclusion Summary (Report.ipynb)

By analyzing the attributes of personal injury in Montgomery county (MD) traffic accidents. We built a Random forest model with a max depth of 19 to predict personal injury in accidents. The features I used for this model are race, agency location, hour, day of week and month. The accuracy of the test model is 81.07%.

## conclusion recommendations (Report.ipynb)

1. In the original data, the traffic violation description part has too many different variables (around 16000+). It's very hard to organize the description. But I think this is a very important feature for predicting accidents and personal injury. Creating more categories for separating violations will be very helpful.


2. Montgomery county's traffic violation data doesn't have the age information. A lot more other places' records do have the age columns. I guess it's an important attribute and wish they can add age to the data.


3. For the accident location, it will be better to have a city column instead of having detailed address. In this report, I use the race column to kind of estimating the location since some places do have more volume of a certain race. But I don't think this method is accurate.

## conclusion next steps (Report.ipynb)

1. I would like to organize the description into different categories to see if will help my model perform better.


2. Personal injury in the accident is not the only outcome of traffic violations. I would like to analyze more outcomes. Explore more different target variables from this data.

## no errors (Report.ipynb)