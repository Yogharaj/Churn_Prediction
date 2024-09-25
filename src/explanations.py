import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import joblib

def explain_with_shap(X_test, dataset_name):
    model = joblib.load(f'models/churn_model_{dataset_name}.pkl')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test)
    return shap_values

def explain_with_lime(X_test, dataset_name):
    model = joblib.load(f'models/churn_model_{dataset_name}.pkl')
    explainer = LimeTabularExplainer(X_test, feature_names=X_test.columns, class_names=['no_churn', 'churn'])
    
    exp = explainer.explain_instance(X_test.iloc[0], model.predict_proba)
    exp.show_in_notebook(show_table=True)
    return exp
