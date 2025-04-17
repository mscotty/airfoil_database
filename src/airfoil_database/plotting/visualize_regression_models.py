import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def visualize_regression_models(df, 
                                response_var, 
                                simple_models, 
                                mult_model, 
                                parsimonious_model,
                                output_dir=None):
    """
    Generates plots to compare:
    - Simple Linear Regression (for each predictor in simple_models)
    - Multiple Linear Regression
    - Stepwise Parsimonious Regression
    - Combined comparison plot

    Parameters:
    df (pd.DataFrame): The dataset.
    response_var (str): The target variable.
    simple_models (dict): Dictionary with predictor variable names as keys and regression models as values.
    mult_model (sm.OLS): The full multiple linear regression model.
    parsimonious_model (dict): Dictionary containing the final parsimonious model and selected variables.
    """

    num_simple_models = len(simple_models)
    fig, axes = plt.subplots(num_simple_models, 2, figsize=(12, num_simple_models * 3.5), gridspec_kw={'height_ratios': [1] * num_simple_models})
    plt.subplots_adjust(hspace=0.4)

    # Define colorblind-friendly colormap
    colorblind_colors = sns.color_palette("colorblind")

    # Ensure axes are always iterable
    if num_simple_models == 1:
        axes = [axes]

    # --- Simple Linear Regression Plots ---
    simple_r2_values = {}  # Store R² values for the combined plot
    for i, (predictor, model) in enumerate(simple_models.items()):
        ax_scatter = axes[i][0]
        ax_residual = axes[i][1]

        # Scatter plot with regression line
        sns.scatterplot(x=df[predictor], y=df[response_var], ax=ax_scatter, label="Actual Data", color="black")
        pred_values = model.predict(sm.add_constant(df[predictor]))
        r2_value = model.rsquared
        simple_r2_values[predictor] = r2_value  # Store for later use
        ax_scatter.plot(df[predictor], pred_values, color=colorblind_colors[i], label=f"Regression Line (R²={r2_value:.2f})", linewidth=2)
        ax_scatter.set_title(f"Simple Linear Regression: {response_var} vs {predictor}")
        ax_scatter.set_xlabel(predictor)
        ax_scatter.set_ylabel(response_var)
        ax_scatter.legend()
        ax_scatter.grid(True, linestyle="--", alpha=0.6)

        # Residual plot
        residuals = df[response_var] - pred_values
        ax_residual.scatter(pred_values, residuals, color=colorblind_colors[i], edgecolor='black', alpha=0.7)
        ax_residual.axhline(0, color='black', linestyle='--')
        ax_residual.set_xlabel("Predicted Values")
        ax_residual.set_ylabel("Residuals")
        ax_residual.set_title(f"Residuals: {response_var} vs {predictor}")
        ax_residual.grid(True, linestyle="--", alpha=0.6)

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'simple_regression_models.png'))
    else:
        plt.show()

    # --- Multiple Linear Regression: Actual vs Predicted ---
    mult_pred = mult_model.predict(sm.add_constant(df[mult_model.model.exog_names[1:]]))
    r2_mult = mult_model.rsquared
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df[response_var], y=mult_pred, label=f"Multiple Regression (R²={r2_mult:.2f})", color=colorblind_colors[2])
    plt.plot(df[response_var], df[response_var], color='black', linestyle="dashed", label="Perfect Fit")
    plt.title("Multiple Linear Regression: Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'multi_lin_reg_model.png'))
    else:
        plt.show()

    # --- Stepwise Parsimonious Regression: Actual vs Predicted ---
    pars_pred = parsimonious_model['final_model'].predict(sm.add_constant(df[parsimonious_model['used_vars']]))
    r2_pars = parsimonious_model['final_model'].rsquared
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=df[response_var], y=pars_pred, label=f"Parsimonious Regression (R²={r2_pars:.2f})", color=colorblind_colors[4])
    plt.plot(df[response_var], df[response_var], color='black', linestyle="dashed", label="Perfect Fit")
    plt.title("Parsimonious Regression: Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'parsimonious_reg_model.png'))
    else:
        plt.show()

    # --- Combined Model Comparison ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[response_var], y=mult_pred, label=f"Multiple Regression (R²={r2_mult:.2f})", color=colorblind_colors[2], marker="o")
    sns.scatterplot(x=df[response_var], y=pars_pred, label=f"Parsimonious Regression (R²={r2_pars:.2f})", color=colorblind_colors[4], marker="s")

    # Add simple regression models dynamically with R² in labels
    for i, (predictor, model) in enumerate(simple_models.items()):
        pred_values = model.predict(sm.add_constant(df[predictor]))
        r2_value = simple_r2_values[predictor]
        sns.scatterplot(x=df[response_var], y=pred_values,
                        label=f"Simple ({predictor}) (R²={r2_value:.2f})", color=colorblind_colors[i], marker="D")

    plt.plot(df[response_var], df[response_var], color='black', linestyle="dashed", label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Model Comparisons")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'combined_reg_model.png'))
    else:
        plt.show()
