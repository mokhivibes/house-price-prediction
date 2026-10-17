from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes(exclude='number').columns.tolist()
    numeric_cols.remove('MSSubClass')
    categorical_cols.append('MSSubClass')

    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                 ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='None')),
                                     ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_pipeline, numeric_cols),
                                      ('cat', categorical_pipeline, categorical_cols)])
    return preprocessor