from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

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



def evaluate(model, X, y):
    pipe = Pipeline([('prep', make_preprocessor(X)), ('model', model)])

    mae_scores  = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
    r2_scores   = cross_val_score(pipe, X, y, cv=5, scoring='r2')

    return -mae_scores.mean(), mae_scores.std(),-rmse_scores.mean(), r2_scores.mean()