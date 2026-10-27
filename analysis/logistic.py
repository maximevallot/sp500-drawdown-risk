from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def logistic(X_train, y_train, X_now):

    if y_train.nunique() < 2:
        return float(y_train.mean())

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    # Predict probability of Event A = 1

    return float(model.predict_proba(X_now)[0,1])