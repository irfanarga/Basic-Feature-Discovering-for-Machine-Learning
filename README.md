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

### Feature Engineering
The first feature/column created is Family_Size, which is a combination of Parent, Children, Sibling, and Spouse. Then add 1 assuming that person counts himself too.

The second feature/column that is created is to combine Family_Size with its respective groups depending on the number.

The categories are as follows:
- Family Size 1 = Alone
- Family Size 2, 3, and 4 = Small
- Family Size 5 and 6 = Medium
- Family Size 7, 8 and 11 = Large
