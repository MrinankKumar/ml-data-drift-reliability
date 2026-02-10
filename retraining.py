from sklearn.metrics import accuracy_score

def retrain_and_evaluate(model, X_new, y_new, X_test, y_test):

    model.fit(X_new, y_new)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc
