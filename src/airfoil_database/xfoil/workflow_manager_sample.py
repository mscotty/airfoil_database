import os

from airfoil_database.xfoil.workflow_manager import XFoilWorkflowManager
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.utilities.get_top_level_module import get_project_root

# Initialize
db = AirfoilDatabase("airfoils.db", os.path.join(get_project_root(), "airfoil_database", ))
workflow = XFoilWorkflowManager(db, "xfoil")

# Define conditions
reynolds_list = [1e6, 2e6]
mach_list = [0.1, 0.2]
alpha_list = [-5, 0, 5, 10, 15]
ncrit_list = [2.0]

# Check status first
status = workflow.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
print(f"Missing {status['total_conditions_missing']} conditions")

# Run analysis
results = workflow.run_batch_analysis(
    reynolds_list=reynolds_list,
    mach_list=mach_list,
    alpha_list=alpha_list,
    ncrit_list=ncrit_list,
    max_workers=6,
    skip_existing=True
)
