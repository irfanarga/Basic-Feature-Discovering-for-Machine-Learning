# Basic Feature Discovering for Machine Learning
The goals are comparing the model with featuring engineering and without featuring engineering

### Using the library:
- Pandas
- Matplotlib
- Seaborn
- Sklearn

### Dataset
In this dataset, there is a Survived column as a target variable. All other columns/features will be used to determine whether this passenger survived the Titanic incident.

### Data cleansing and Correlation
Filling Missing Value

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Missing%20value.png)

There are some missing value in the Age, Cabin, and Embarked column

- Missing value in the Age data is filled with the median value of Age data based on passenger class (Pclass) and Sex
- For Embarked, most of the people from Titanic depart from Southampton/S, so we can fill it with S.
- Missing value in the Fare data is filled with the median value of Fare data based on passenger class (Pclass), Parch, and SibSp

Correlation

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Correlation.png)

The correlation results show that the target variable (Survived) has a very large correlation with PClass and Fare. Meanwhile, Age is closely related to Pclass, Sibling Spouse (SibSp), Parent Children (Parch). It can be assumed that most of the survivors are people with upper Pclass and a person's parents can say he will bring siblings/parents/children/spouse. And Fare (price) is of course related to the Pclass (passenger class) of a passenger.

### Visualization

![]

The graph shows that passengers departing from Cherbourg were more likely to survive, while passengers from Southampton, only half survived. For people carrying only 1 Parent / Child, more survivors. For Passenger Class 1, the chances of survival are much higher. And only a few passengers with Passenger Class 3 survived. And the person carrying 1 Sibling / Spouse is much more likely to survive. The person carrying 2 Siblings/ Spouses is quite unlikely to survive.
