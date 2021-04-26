from sklearn.linear_model import LinearRegression

#BUILD MODEL HERE
model = LinearRegression()

def get_predictions(test_case, X, Y):
    model.fit(X, Y)
    y_pred = model.predict(test_case)
    return round(y_pred[0][0])
