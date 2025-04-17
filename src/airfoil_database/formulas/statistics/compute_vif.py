import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df, selected_vars):
    """Computes Variance Inflation Factor (VIF) to check multicollinearity."""
    if len(selected_vars) < 2:
        return pd.DataFrame({"Variable": selected_vars, "VIF": [1.0]})  # Single variable has VIF=1
    
    X = sm.add_constant(df[selected_vars])  # Add intercept
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data[vif_data["Variable"] != "const"]  # Exclude intercept

def test():
    # ✅ **Test with a Sample Dataset**
    test_data = pd.DataFrame({
        "mpg": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "cyl": [4, 4, 6, 6, 8, 8, 4, 6, 8, 4],
        "disp": [160, 160, 258, 258, 360, 360, 140, 200, 320, 180],
        "wt": [2.62, 2.88, 3.21, 3.44, 3.57, 3.78, 2.46, 3.00, 3.68, 2.80]
    })

    selected_vars = ["cyl", "disp", "wt"]
    vif_report = compute_vif(test_data, selected_vars)
    print("\n✅ **VIF Report**")
    print(vif_report)


