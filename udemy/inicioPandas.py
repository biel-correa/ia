import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
df = pd.read_csv('/home/gabriel/Documentos/github/udemy/arquivos/titanic-train.csv')
# df.head()
# df.info()
# df.describe()

# print(df.iloc[665])
# df.loc[0:4,'Ticket']
# df['Ticket'].head()
# df[['Embarked','Ticked']].head()


# df['Age']>70
# df[(df['Age'] == 11) & (df['SibSp'] == 5)]
# df[(df.Age == 11) | (df.SibSp == 5)]
# df.query('(Age ==11) | (SibSp == 5)')
# print(df[df['Age']>70])

# print(df['Embarked'].unique())

# print(df.sort_values('Age',ascending = True).head())


#criar tabela com informacao especifica
# print(df.pivot_table(
#     index='Pclass',
#     columns='Survived',
#     values='PassengerId',
#     aggfunc='count'
# ))

# print(df.groupby(['Pclass','Survived'])['PassengerId'].count())
# correlatedWithSurvived = df.corr()['Survived'].sort_values()
# print(correlatedWithSurvived)