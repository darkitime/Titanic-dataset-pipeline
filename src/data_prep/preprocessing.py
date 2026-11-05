import pandas as pd
import numpy 

def clean(df, is_one_hot_encoding = False):
    '''
    Fill Nan vaules, 
    convert str to num (is_one_hot_encoding = False)
    '''
    data = df.copy()
    
    # Title replacing
    data['Initial'] = data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
    mapping = {
        'Mlle': 'Miss', 'Mme': 'Miss', 'Ms': 'Miss',
        'Dr': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Sir': 'Mr', 'Don': 'Mr',
        'Lady': 'Mrs', 'Countess': 'Mrs', 'Dona': 'Mrs',
        'Jonkheer': 'Other', 'Col': 'Other', 'Rev': 'Other'
    }
    data['Initial'] = data['Initial'].replace(mapping)
    data['Initial'] = data['Initial'].fillna('Other')
    # Fill Nan in Age
    age_medians = {'Mr': 33, 'Mrs': 36, 'Master': 5, 'Miss': 22, 'Other': 46}
    for title, age in age_medians.items():
        data.loc[(data['Age'].isnull()) & (data['Initial'] == title), 'Age'] = age
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # Fill Nan in Embarked and Fare
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Fare'] = data['Fare'].fillna(13.3071)

    #Encdoing gender to number
    data['Sex'] = (data['Sex'] == 'female').astype(int)


    #One-hot-encoding
    if is_one_hot_encoding:
        data = pd.get_dummies(data, columns=['Embarked', 'Initial'], drop_first=True, dtype=float)
    else:
        embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        initial_map = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4}
        
        data['Embarked'] = data['Embarked'].map(embarked_map).fillna(0).astype(float)
        data['Initial'] = data['Initial'].map(initial_map).fillna(4).astype(float)

        cat_cols = ['Embarked', 'Initial']
        num_cols = [c for c in data.columns if c not in cat_cols and c not in ['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived']]

    data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, errors='ignore')

    return data

