# from pathlib import Path
#
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn import linear_model as model
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
#
# # Path of file
# path = Path("data/houses.csv")
#
# df = pd.read_csv(path)
#
# # Assign X and Y axis
# X = df[['GarageArea', 'GrLivArea']]#.apply(pd.to_numeric)
# y = df[['SalePrice']]#.apply(pd.to_numeric)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
#
# print(r2_score(y_test, y_pred))
# #
# # y_pred = regressor.predict(X_test)
# #
# # print(r2_score(y_test, y_pred))
# # print(X_test['GarageArea'])
# #
# # fig = plt.figure(figsize=(10, 8))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X['GarageArea'], X['YearBuilt'], y, c='blue', marker='o')
# # #ax.plot(X_test['GarageArea'], X_test['YearBuilt'], y_pred, color='blue', linewidth=3)
# # # set your labels
# # ax.set_xlabel('Garage Area')
# # ax.set_ylabel('Year Built')
# # ax.set_zlabel('Price')
# #
# # plt.show()
