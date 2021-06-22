from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # From https://deep-forest.readthedocs.io/en/latest/index.html 
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    esti = [1,10,20,30]
    MSEs = []
    for i in esti:
        model = CascadeForestRegressor(n_estimators=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        MSEs.append(mse)
        print("\nTesting MSE: {:.3f}".format(mse))
    plt.xlabel("n_estimators")
    plt.ylabel("MSE")
    plt.plot(esti, MSEs)
    plt.show()

    model = CascadeForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    MSEs.append(mse)
    print("\nDF21 Testing MSE: {:.3f}".format(mse))

    RFmodel = RandomForestRegressor()
    RFmodel.fit(X_train, y_train)
    y_pred = RFmodel.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("\nRF Testing MSE: {:.3f}".format(mse))

