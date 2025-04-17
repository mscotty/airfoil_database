import pandas as pd
import statsmodels.api as sm


def linear_regression_model(df, response_var, predictor_vars):
    # Prepare table to store results
    results = []

    # Loop through each predictor and fit a simple linear regression model
    for predictor in predictor_vars:
        X = df[[predictor]]  # Predictor variable
        X = sm.add_constant(X)  # Add intercept term
        y = df[response_var]  # Response variable
        
        model = sm.OLS(y, X).fit()  # Fit regression model
        
        # Extract required values
        beta0 = model.params["const"]  # Intercept
        beta1 = model.params[predictor]  # Slope
        t_stat = model.tvalues[predictor]  # t-stat for β1
        lcl, ucl = model.conf_int().loc[predictor]  # Confidence Interval for β1
        r2 = model.rsquared  # R² value
        
        # Append to results list
        results.append([predictor, beta0, beta1, t_stat, lcl, ucl, r2])

    # Convert to DataFrame
    return pd.DataFrame(results, columns=["Predictor", "β0", "β1", "t-stat", "LCL", "UCL", "R2"])
