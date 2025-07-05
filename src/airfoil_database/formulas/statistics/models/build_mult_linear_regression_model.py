import statsmodels.api as sm

def multiple_linear_regression(df, response_var, predictor_vars, display_summary=True):
    """
    Fits a multiple linear regression model and returns the summary.

    Parameters:
    df (pd.DataFrame): The dataset containing the target and predictor_vars.
    target_var (str): The name of the response variable (dependent variable).
    predictor_vars (list): A list of predictor variable names (independent variables).
    display_summary (bool): Whether to print the summary (default is True).

    Returns:
    sm.OLS: The fitted regression model.
    """
    # Ensure all predictor_vars exist in the DataFrame
    X = df[predictor_vars]

    # Add intercept term
    X = sm.add_constant(X)

    # Define response variable
    y = df[response_var]

    # Fit multiple linear regression model
    model = sm.OLS(y, X).fit()

    # Display or return the summary
    if display_summary:
        print(model.summary())

    return model  # Returning the model object for further use

