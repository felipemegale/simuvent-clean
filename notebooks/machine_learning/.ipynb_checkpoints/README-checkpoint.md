# ML methods I will evaluate

## My class of problem is regression because I need to predict values in a time series

### Metrics used to evaluate:
- R-Squared
- Mean Absolute Error
- Mean Absolute Percentage Error
- Mean Squared Error
- Root Mean Squared Error
- Max Error
- Explained Variance

### To find best hyperparameters for my models
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Things I might use:
- [Meta-estimator to regress on a transformed target](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
- [Cross-decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html)
- [PLS Regression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
- [Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [PLSCanonical](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html)
- Ensemble Methods:
    - [An AdaBoost regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
    - [A Bagging regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
    - [An extra-trees regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
    - [Gradient Boosting for regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    - [A random forest regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - [Stack of estimators with a final regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
    - [Histogram-based Gradient Boosting Regression Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Gaussian process regression (GPR)](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
    - Will not be used because it worsens when the number of features increase
- [Kernel ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
    - Could not use Kernel Ridge due to high memory demand
- Linear methods:
    - [Ordinary least squares Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    - [Ridge regression with built-in cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
    - [Linear model fitted by minimizing a regularized empirical loss with SGD](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
    - [Elastic Net model with iterative fitting along a regularization path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
    - [Cross-validated Least Angle Regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html)
    - [Lasso linear model with iterative fitting along a regularization path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
    - [Cross-validated Lasso, using the LARS algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html)
    - [Lasso model fit with Lars using BIC or AIC for model selection](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html)
- Bayesian regressor:
    - [Bayesian ARD regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
- Outlier-robust regressors:
    - [L2-regularized linear regression model that is robust to outliers](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
    - [Linear regression model that predicts conditional quantiles](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)
    - [Theil-Sen Estimator: robust multivariate regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html)
- Generalized linear models (GLM) for regression:
    - [Generalized Linear Model with a Poisson distribution](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
    - [Generalized Linear Model with a Tweedie distribution](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)
    - [Generalized Linear Model with a Gamma distribution](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html)
- Support Vector Machines:
    - [Linear Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)
    - [Nu Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)
    - [Epsilon-Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- Decision Trees:
    - [A decision tree regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    - [An extremely randomized tree regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)