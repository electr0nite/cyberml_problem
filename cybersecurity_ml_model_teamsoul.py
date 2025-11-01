import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import textwrap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score
import warnings
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = "Global_Cybersecurity_Threats_2015-2024.csv"

# These are the *inputs* for the original models
ORIGINAL_FEATURES = [
    'Country',
    'Attack Type',
    'Target Industry',
    'Attack Source',
    'Security Vulnerability Type'
]

# Define features for the new task (predicting Attack Source)
# We must remove 'Attack Source' from the list of inputs.
ATTACK_SOURCE_PREDICTION_FEATURES = [
    'Country',
    'Attack Type',
    'Target Industry',
    'Security Vulnerability Type'
]

# Define all the prediction tasks
TARGET_CONFIGS = {
    '1': {
        'name': 'Financial Loss (in Million $)',
        'type': 'regression',
        'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'features': ORIGINAL_FEATURES
    },
    '2': {
        'name': 'Number of Affected Users',
        'type': 'regression',
        'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'features': ORIGINAL_FEATURES
    },
    '3': {
        'name': 'Defense Mechanism Used',
        'type': 'classification',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'features': ORIGINAL_FEATURES
    },
    '4': {
        'name': 'Incident Resolution Time (in Hours)',
        'type': 'regression',
        'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'features': ORIGINAL_FEATURES
    },
    '5': {
        'name': 'Attack Source',
        'type': 'classification',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'features': ATTACK_SOURCE_PREDICTION_FEATURES
    }
}

def load_data(csv_path):
    """Loads and cleans the dataset."""
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None

    print("Dataset loaded successfully.")

    df.columns = df.columns.str.strip()

    print("Cleaning whitespace from data...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Check if all features and targets exist
    all_target_names = set(c['name'] for c in TARGET_CONFIGS.values())
    all_feature_names = set()
    for config in TARGET_CONFIGS.values():
        all_feature_names.update(config['features'])

    all_required_cols = all_target_names | all_feature_names # Combine all possible columns

    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing: {missing_cols}")
        return None

    return df

# --- Plotting Functions ---

def plot_regression_performance(y_test, predictions, target_name, mae, r2):
    """Saves a scatter plot for regression models."""
    plt.figure(figsize=(12, 7))
    sns.regplot(x=y_test, y=predictions,
                scatter_kws={'alpha':0.3, 'color': 'blue'},
                line_kws={'color':'red', 'linestyle':'--', 'linewidth': 2})

    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

    plt.title(f"Model Performance: Actual vs. Predicted {target_name}", fontsize=16, fontweight='bold')
    plt.xlabel(f"Actual {target_name}", fontsize=12)
    plt.ylabel(f"Predicted {target_name}", fontsize=12)

    metrics_text = f"R² Score: {r2:.3f}\nMAE: {mae:.2f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.legend()
    plt.grid(True)

    plot_filename = 'model_performance.png'
    plt.savefig(plot_filename)
    print(f"\n[GRAPH] Regression performance plot saved to '{plot_filename}'")
    plt.close()

def plot_classification_performance(y_test, predictions, model):
    """Saves a confusion matrix for classification models."""
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 9))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
    plt.ylabel("Actual Value", fontsize=12)
    plt.xlabel("Predicted Value", fontsize=12)

    plot_filename = 'confusion_matrix.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\n[GRAPH] Classification confusion matrix saved to '{plot_filename}'")
    plt.close()

def plot_feature_importance(model, feature_names):
    """Saves a bar chart of the top 20 most important features."""
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    top_20 = forest_importances.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_20.values, y=top_20.index, palette="vlag")

    plt.title("Top 20 Most Important Features", fontsize=16, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    plot_filename = 'feature_importance.png'
    plt.savefig(plot_filename)
    print(f"[GRAPH] Feature importance plot saved to '{plot_filename}'")
    plt.close()

# --- Core Model Functions ---

def train_and_evaluate_model(df, target_name, model, model_type, feature_list):
    """Trains a model for the chosen target and evaluates it."""
    print(f"\n--- Training Model to Predict: '{target_name}' ---")

    X = df[feature_list]
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")

    preprocessor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    processed_feature_names = preprocessor.get_feature_names_out(feature_list)

    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train_processed, y_train)
    print("Model training complete.")

    predictions = model.predict(X_test_processed)

    # --- Evaluate Based on Model Type ---
    print("\n--- Model Evaluation (on Test Data) ---")
    if model_type == 'regression':
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R2) Score: {r2:.3f}")
        plot_regression_performance(y_test, predictions, target_name, mae, r2)

    elif model_type == 'classification':
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy Score: {acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        plot_classification_performance(y_test, predictions, model)
    print("---------------------------------------")

    plot_feature_importance(model, processed_feature_names)

    return model, preprocessor, X_train_processed, processed_feature_names

def get_user_scenario(df, feature_list):
    """Interactively asks the user for a new scenario."""
    print("\n--- Create a New Attack Scenario ---")
    print("Enter the parameters for the attack. Type '?' to see all options.")

    user_input = {}

    for feature in feature_list:
        options = sorted(df[feature].unique())
        options_lower_map = {opt.lower(): opt for opt in options}

        while True:
            prompt = f"Enter value for '{feature}': "
            val = input(prompt).strip()

            if val == '?':
                print(f"\nOptions for '{feature}':")
                print(textwrap.fill(", ".join(options), width=80))
                print("-" * 20)
                continue

            if val.lower() in options_lower_map:
                correct_val = options_lower_map[val.lower()]
                user_input[feature] = correct_val
                if val != correct_val:
                    print(f"   > Matched: '{correct_val}'")
                break
            else:
                print(f"Invalid input: '{val}'. Type '?' for options or try again.")

    return pd.DataFrame([user_input])

def explain_prediction(model, model_type, preprocessor, X_train_processed_df, scenario_df, feature_names):
    """Uses SHAP to explain a single prediction."""
    print("\n--- Generating Prediction Explanation ---")

    explainer = shap.TreeExplainer(model)

    scenario_processed = preprocessor.transform(scenario_df)
    scenario_processed_df = pd.DataFrame(scenario_processed, columns=feature_names)

    shap_values = explainer.shap_values(scenario_processed_df)

    plot_filename = 'prediction_explanation.png'
    plt.figure()

    if model_type == 'regression':
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[0],
            data=scenario_processed_df.iloc[0],
            feature_names=feature_names
        )
        print(f" - Base average prediction: {explainer.expected_value[0]:.2f}")

    elif model_type == 'classification':
        # Get the predicted class name
        predicted_class_name = model.predict(scenario_processed)[0]
        # Get the integer index of that class
        predicted_class_index = list(model.classes_).index(predicted_class_name)

        print(f" - Explaining prediction for class: '{predicted_class_name}'")
        print(f" - Base average log-odds: {explainer.expected_value[predicted_class_index]:.2f}")

        # Select the SHAP values *for that specific class*
        explanation = shap.Explanation(
            values=shap_values[predicted_class_index][0],
            base_values=explainer.expected_value[predicted_class_index],
            data=scenario_processed_df.iloc[0],
            feature_names=feature_names
        )

    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\n[GRAPH] Prediction explanation plot saved to '{plot_filename}'")
    print(" - Red bars pushed the prediction HIGHER (or more likely).")
    print(" - Blue bars pushed the prediction LOWER (or less likely).")
    plt.close()

# --- Main Execution ---
def main():
    sns.set_theme(style="whitegrid", palette="muted")

    df = load_data(DATA_FILE)
    if df is None:
        return

    while True:
        # --- Main Menu ---
        print("\n" + "="*50)
        print(" CHOOSE A PREDICTION TASK:")
        print("="*50)
        for key, config in TARGET_CONFIGS.items():
            print(f" {key}: Predict {config['name']} ({config['type']})")
        print(" 6: Exit")
        print("-"*50)

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '6':
            print("Exiting. Thank you!")
            break

        if choice not in TARGET_CONFIGS:
            print("Invalid choice. Please try again.")
            continue

        # --- Run the Chosen Task ---
        config = TARGET_CONFIGS[choice]
        target_name = config['name']
        model_type = config['type']
        model = config['model']
        feature_list = config['features'] # Get the specific features for this task

        try:
            trained_model, preprocessor, X_train_processed, p_feature_names = \
                train_and_evaluate_model(df, target_name, model, model_type, feature_list)

            X_train_processed_df = pd.DataFrame(X_train_processed, columns=p_feature_names)

            # --- Interactive Loop for this task ---
            while True:
                scenario_df = get_user_scenario(df, feature_list)

                scenario_processed = preprocessor.transform(scenario_df)
                predicted_value = trained_model.predict(scenario_processed)[0]

                print("\n--- Prediction Result ---")
                print("For the scenario:")
                print(scenario_df.to_string(index=False))
                print("-------------------------------------")

                if model_type == 'regression':
                    unit = "Users" if "Users" in target_name else ("Hours" if "Hours" in target_name else "Million $")
                    print(f"--> Predicted {target_name}: {predicted_value:,.2f} {unit}")
                else:
                    print(f"--> Predicted {target_name}: {predicted_value}")
                print("-------------------------------------")

                explain_prediction(trained_model, model_type, preprocessor, X_train_processed_df, scenario_df, p_feature_names)

                print("\n" + "="*50)
                again = input(f"Predict another '{target_name}' scenario? (yes/no): ").strip().lower()
                if again not in ['yes', 'y']:
                    break # Go back to the main menu

        except Exception as e:
            print(f"\nAn error occurred during the process: {e}")
            print("Please check your data and try again.")
            # import traceback
            # traceback.print_exc() # Uncomment for deep debugging

if __name__ == "__main__":
    main()