from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from data.dataset import get_corpus

TFIDF_PATH = Path("saves") / "LogisticRegression" / "tfidf.pkl"
REGRESSION_PATH = Path("saves") / "LogisticRegression" / "regression.pkl"

class LogisticRegressionModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1,2),
                                     min_df=2,
                                     max_df=0.9)
        self.regression = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        TFIDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        X = None
        try:
            self.tfidf = joblib.load(TFIDF_PATH)
            X = self.tfidf.transform(X_train)
        except Exception:
            self.tfidf = TfidfVectorizer()
            X = self.tfidf.fit_transform(X_train)
            joblib.dump(self.tfidf, TFIDF_PATH)

        try:
            self.regression = joblib.load(REGRESSION_PATH)
        except Exception:
            self.regression.fit(X, y_train)
            joblib.dump(self.regression, REGRESSION_PATH)

    def predict(self, X):
        X = self.tfidf.transform(X)
        return self.regression.predict(X)

    def predict_proba(self, X):
        X = self.tfidf.transform(X)
        return self.regression.predict_proba(X)[:, 1]


