import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Open all dataset and merged
def concat_df(train_data, test_data):
	# Return a concatenated df of training and test set
	return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

df_train = pd.read_csv('data_train.csv')
df_test = pd.read_csv('data_test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

#Explore row and column numbers
print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))
print(df_train.columns)
print(df_test.columns)

#Correlation features
df_train_corr = df_train.corr().abs()
print(df_train_corr.to_string())

#Recognize missing value in column
def display_missing(df):
	for col in df.columns.tolist():
		print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
	print('\n')

for df in dfs:
	print('{}'.format(df.name))
	display_missing(df)

#Fill missing value of Age, Embarked and Fare
#Filling Age values
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
	for sex in ['female', 'male']:
		print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
	print('Median age of all passengers: {}'.format(df_all['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#Filling Embarked values
# Filling the missing values in Embarked with S
df_all['Embarked'] = df_all['Embarked'].fillna('S')

#Filling Fare values
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df_all['Fare'] = df_all['Fare'].fillna(med_fare)
