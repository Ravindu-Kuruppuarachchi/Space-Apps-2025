import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class ExoplanetEnsembleClassifier:
    """
    Ensemble classification system for exoplanet detection.
    Tests multiple ensemble algorithms and compares their performance.
    """
    
    def __init__(self, train_path='data/train.csv', 
                 val_path='data/validation.csv', 
                 test_path='data/test.csv'):
        """
        Initialize the classifier with data paths.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the train, validation, and test datasets.
        """
        print("="*70)
        print("LOADING AND PREPROCESSING DATA")
        print("="*70)
        
        # Load datasets
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        
        print(f"\nTrain set: {train_df.shape}")
        print(f"Validation set: {val_df.shape}")
        print(f"Test set: {test_df.shape}")
        
        # Separate features and target
        # Remove non-feature columns: kepid, koi_disposition
        feature_cols = [col for col in train_df.columns 
                       if col not in ['kepid', 'koi_disposition']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['koi_disposition']
        
        X_val = val_df[feature_cols]
        y_val = val_df['koi_disposition']
        
        X_test = test_df[feature_cols]
        y_test = test_df['koi_disposition']
        
        print(f"\nNumber of features: {len(feature_cols)}")
        print(f"\nClass distribution in training set:")
        print(y_train.value_counts())
        
        # Encode labels (CONFIRMED=1, FALSE POSITIVE=0)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                 self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"\nLabel encoding: {label_mapping}")
        
        # Handle missing values
        print("\nHandling missing values with median imputation...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Scale features
        print("Scaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        print("\nData preprocessing completed!")
        
        return (X_train_scaled, y_train_encoded, 
                X_val_scaled, y_val_encoded,
                X_test_scaled, y_test_encoded)
    
    def initialize_models(self):
        """
        Initialize all ensemble classification models.
        """
        print("\n" + "="*70)
        print("INITIALIZING ENSEMBLE MODELS")
        print("="*70)
        
        self.models = {
            'Naive Bayes': GaussianNB(),
            
            'K-Nearest Neighbor': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            
            'XGBoost Classifier': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42
            ),
            
            'LGBM Classifier': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            ),
            
            'Bagging Classifier': BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=100,
                random_state=42
            )
        }
        
        print(f"\nInitialized {len(self.models)} classification models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train all models and evaluate their performance.
        """
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Print classification report
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=self.label_encoder.classes_))
        
        print("\n" + "="*70)
        print("ALL MODELS TRAINED AND EVALUATED")
        print("="*70)
    
    def plot_accuracy_comparison(self, save_path='model_accuracy_comparison.png'):
        """
        Create a bar plot comparing accuracies of all models.
        """
        print("\n" + "="*70)
        print("GENERATING ACCURACY COMPARISON PLOT")
        print("="*70)
        
        # Sort models by accuracy
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['accuracy'])
        
        model_names = [item[0] for item in sorted_results]
        accuracies = [item[1]['accuracy'] * 100 for item in sorted_results]
        
        # Create figure
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(model_names)), accuracies, 
                       color='#3498db', edgecolor='black', linewidth=1.2)
        
        # Customize plot
        plt.xlabel('Classification Algorithm', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title('Ensemble Classification Algorithms - Exoplanet Detection Accuracy', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        plt.text(0.02, 0.98, '■ Accuracy', transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.show()
        
        # Print accuracy table
        print("\n" + "="*70)
        print("ACCURACY SUMMARY TABLE")
        print("="*70)
        print(f"{'Rank':<6} {'Model Name':<25} {'Accuracy':<12}")
        print("-"*70)
        for rank, (name, acc) in enumerate(zip(model_names, accuracies), 1):
            print(f"{rank:<6} {name:<25} {acc:>10.2f}%")
        print("="*70)
    
    def plot_confusion_matrices(self, top_n=3, save_path='confusion_matrices.png'):
        """
        Plot confusion matrices for top N performing models.
        """
        print("\n" + "="*70)
        print(f"GENERATING CONFUSION MATRICES FOR TOP {top_n} MODELS")
        print("="*70)
        
        # Get top N models
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
        top_models = sorted_results[:top_n]
        
        # Create subplots
        fig, axes = plt.subplots(1, top_n, figsize=(6*top_n, 5))
        if top_n == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(top_models):
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]*100:.2f}%',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('Actual', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to: {save_path}")
        plt.show()
    
    def save_results(self, output_path='model_results_summary.txt'):
        """
        Save detailed results to a text file.
        """
        print("\n" + "="*70)
        print("SAVING RESULTS SUMMARY")
        print("="*70)
        
        with open(output_path, 'w') as f:
            f.write("EXOPLANET CLASSIFICATION - MODEL COMPARISON RESULTS\n")
            f.write("="*70 + "\n\n")
            
            sorted_results = sorted(self.results.items(), 
                                   key=lambda x: x[1]['accuracy'], 
                                   reverse=True)
            
            for rank, (name, result) in enumerate(sorted_results, 1):
                f.write(f"\nRank {rank}: {name}\n")
                f.write("-"*70 + "\n")
                f.write(f"Accuracy: {result['accuracy']*100:.2f}%\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(result['true_labels'], 
                                             result['predictions'],
                                             target_names=self.label_encoder.classes_))
                f.write("\n" + "="*70 + "\n")
        
        print(f"Results summary saved to: {output_path}")
    
    def run_complete_pipeline(self):
        """
        Run the complete classification pipeline.
        """
        print("\n" + "="*70)
        print("STARTING EXOPLANET ENSEMBLE CLASSIFICATION PIPELINE")
        print("="*70)
        
        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate
        self.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Generate visualizations
        self.plot_accuracy_comparison()
        self.plot_confusion_matrices(top_n=3)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return self.results


if __name__ == "__main__":
    # Create classifier instance
    classifier = ExoplanetEnsembleClassifier(
        train_path='data/train.csv',
        val_path='data/validation.csv',
        test_path='data/test.csv'
    )
    
    # Run complete pipeline
    results = classifier.run_complete_pipeline()
    
    print("\n🚀 Exoplanet classification complete!")
