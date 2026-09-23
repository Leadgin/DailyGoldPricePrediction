import pandas as pd
from sklearn.model_selection import train_test_split
gold = pd.read_csv('Gold Price.csv')

gold = gold[gold['Volume'] > 0]


X_gold= gold.loc[:,'Open':'Chg%']  
Y_gold = gold['Price']


x_train, x_test, y_train, y_test = train_test_split(X_gold, Y_gold,random_state=1) 

x_train.to_csv('x_train.csv', index=False)
x_test.to_csv('x_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)