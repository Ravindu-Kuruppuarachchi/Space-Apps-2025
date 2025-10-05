import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc, 
                            precision_recall_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class ExoplanetDeepNeuralNetwork:
    """
    Deep Neural Network for Exoplanet Detection.
    Multi-layer architecture with dropout, batch normalization, and regularization.
    """
    
    def __init__(self, train_path='data/train.csv', 
                 val_path='data/validation.csv', 
                 test_path='data/test.csv'):
        """
        Initialize the deep neural network classifier.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
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
        feature_cols = [col for col in train_df.columns 
                       if col not in ['kepid', 'koi_disposition']]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        X_train = train_df[feature_cols]
        y_train = train_df['koi_disposition']
        
        X_val = val_df[feature_cols]
        y_val = val_df['koi_disposition']
        
        X_test = test_df[feature_cols]
        y_test = test_df['koi_disposition']
        
        print(f"\nNumber of features: {len(feature_cols)}")
        print(f"\nClass distribution in training set:")
        print(y_train.value_counts())
        
        # Encode labels
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
    
    def build_model(self, input_dim, architecture='deep'):
        """
        Build a deep neural network model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        architecture : str
            Model architecture type: 'deep', 'wide', or 'custom'
        """
        print("\n" + "="*70)
        print("BUILDING DEEP NEURAL NETWORK")
        print("="*70)
        
        model = models.Sequential(name='Exoplanet_DNN')
        
        if architecture == 'deep':
            # Deep architecture with multiple hidden layers
            print("\nArchitecture: Deep Network (6 hidden layers)")
            print("-" * 70)
            
            # Input layer
            model.add(layers.Input(shape=(input_dim,), name='input_layer'))
            
            # Hidden Layer 1
            model.add(layers.Dense(256, name='dense_1'))
            model.add(layers.BatchNormalization(name='bn_1'))
            model.add(layers.Activation('relu', name='relu_1'))
            model.add(layers.Dropout(0.3, name='dropout_1'))
            
            # Hidden Layer 2
            model.add(layers.Dense(128, name='dense_2'))
            model.add(layers.BatchNormalization(name='bn_2'))
            model.add(layers.Activation('relu', name='relu_2'))
            model.add(layers.Dropout(0.3, name='dropout_2'))
            
            # Hidden Layer 3
            model.add(layers.Dense(64, name='dense_3'))
            model.add(layers.BatchNormalization(name='bn_3'))
            model.add(layers.Activation('relu', name='relu_3'))
            model.add(layers.Dropout(0.2, name='dropout_3'))
            
            # Hidden Layer 4
            model.add(layers.Dense(32, name='dense_4'))
            model.add(layers.BatchNormalization(name='bn_4'))
            model.add(layers.Activation('relu', name='relu_4'))
            model.add(layers.Dropout(0.2, name='dropout_4'))
            
            # Hidden Layer 5
            model.add(layers.Dense(16, name='dense_5'))
            model.add(layers.BatchNormalization(name='bn_5'))
            model.add(layers.Activation('relu', name='relu_5'))
            model.add(layers.Dropout(0.1, name='dropout_5'))
            
            # Hidden Layer 6
            model.add(layers.Dense(8, name='dense_6'))
            model.add(layers.Activation('relu', name='relu_6'))
            
            # Output layer
            model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
            
            print("\nLayer Configuration:")
            print(f"  Input Layer: {input_dim} neurons")
            print(f"  Hidden Layer 1: 256 neurons (ReLU + BatchNorm + Dropout 0.3)")
            print(f"  Hidden Layer 2: 128 neurons (ReLU + BatchNorm + Dropout 0.3)")
            print(f"  Hidden Layer 3: 64 neurons (ReLU + BatchNorm + Dropout 0.2)")
            print(f"  Hidden Layer 4: 32 neurons (ReLU + BatchNorm + Dropout 0.2)")
            print(f"  Hidden Layer 5: 16 neurons (ReLU + BatchNorm + Dropout 0.1)")
            print(f"  Hidden Layer 6: 8 neurons (ReLU)")
            print(f"  Output Layer: 1 neuron (Sigmoid)")
        
        elif architecture == 'wide':
            # Wide architecture
            print("\nArchitecture: Wide Network (4 hidden layers)")
            print("-" * 70)
            
            model.add(layers.Input(shape=(input_dim,)))
            
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.4))
            
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
            
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.2))
            
            model.add(layers.Dense(64, activation='relu'))
            
            model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\n" + "="*70)
        print("MODEL COMPILATION COMPLETED")
        print("="*70)
        print(f"Optimizer: Adam (learning_rate=0.001)")
        print(f"Loss Function: Binary Crossentropy")
        print(f"Metrics: Accuracy, Precision, Recall, AUC")
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1):
        """
        Train the deep neural network.
        """
        print("\n" + "="*70)
        print("TRAINING DEEP NEURAL NETWORK")
        print("="*70)
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Training Samples: {len(X_train)}")
        print(f"  Validation Samples: {len(X_val)}")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    
        model_checkpoint = callbacks.ModelCheckpoint(
            'best_exoplanet_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        print("\nCallbacks configured:")
        print("  - Early Stopping (patience=15)")
        print("  - Learning Rate Reduction (patience=5)")
        print("  - Model Checkpoint (saving best model)")
        
        print("\n" + "="*70)
        print("Starting Training...")
        print("="*70 + "\n")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=verbose
        )
        
        self.history = history
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data.
        """
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        # Evaluate
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"  Precision: {test_results[2]:.4f}")
        print(f"  Recall: {test_results[3]:.4f}")
        print(f"  AUC: {test_results[4]:.4f}")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        return y_pred, y_pred_prob
    
    def save_model_and_preprocessors(self, model_dir='saved_model'):
        """
        Save the trained model and all preprocessing objects.
        """
        import os
        
        print("\n" + "="*70)
        print("SAVING MODEL AND PREPROCESSORS")
        print("="*70)
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Save Keras model in H5 format
        h5_path = os.path.join(model_dir, 'exoplanet_model.h5')
        self.model.save(h5_path)
        print(f"\n✓ Keras model saved: {h5_path}")
        
        # 2. Save model in SavedModel format (for TensorFlow Serving)
        savedmodel_path = os.path.join(model_dir, 'exoplanet_savedmodel')
        self.model.save(savedmodel_path, save_format='tf')
        print(f"✓ TensorFlow SavedModel saved: {savedmodel_path}")
        
        # 3. Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved: {scaler_path}")
        
        # 4. Save imputer
        imputer_path = os.path.join(model_dir, 'imputer.pkl')
        with open(imputer_path, 'wb') as f:
            pickle.dump(self.imputer, f)
        print(f"✓ Imputer saved: {imputer_path}")
        
        # 5. Save label encoder
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder saved: {label_encoder_path}")
        
        # 6. Save feature columns
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✓ Feature columns saved: {features_path}")
        
        # 7. Save model metadata
        metadata = {
            'model_name': 'Exoplanet Deep Neural Network',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist(),
            'architecture': 'deep',
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Model metadata saved: {metadata_path}")
        
        print("\n" + "="*70)
        print("ALL COMPONENTS SAVED SUCCESSFULLY!")
        print("="*70)
        print(f"\nSaved files in '{model_dir}' directory:")
        print("  - exoplanet_model.h5 (Keras model)")
        print("  - exoplanet_savedmodel/ (TensorFlow SavedModel)")
        print("  - scaler.pkl (StandardScaler)")
        print("  - imputer.pkl (SimpleImputer)")
        print("  - label_encoder.pkl (LabelEncoder)")
        print("  - feature_columns.pkl (Feature names)")
        print("  - model_metadata.json (Model information)")
        
        return model_dir
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training and validation metrics over epochs.
        """
        print("\n" + "="*70)
        print("GENERATING TRAINING HISTORY PLOTS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Deep Neural Network Training History', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix.
        """
        print("\n" + "="*70)
        print("GENERATING CONFUSION MATRIX")
        print("="*70)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Deep Neural Network\nExoplanet Detection', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = accuracy_score(y_test, y_pred)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy*100:.2f}%', 
                ha='center', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_prob, save_path='roc_curve.png'):
        """
        Plot ROC curve.
        """
        print("\n" + "="*70)
        print("GENERATING ROC CURVE")
        print("="*70)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', linewidth=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Deep Neural Network\nExoplanet Detection', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curve saved to: {save_path}")
        plt.show()
    
    def print_model_summary(self):
        """
        Print detailed model summary.
        """
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70 + "\n")
        self.model.summary()
        
        # Calculate total parameters
        trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        
        print("\n" + "="*70)
        print(f"Total Parameters: {trainable_params + non_trainable_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print("="*70)
    
    def run_complete_pipeline(self, epochs=100, batch_size=32, architecture='deep'):
        """
        Run the complete deep learning pipeline.
        """
        print("\n" + "="*70)
        print("STARTING DEEP NEURAL NETWORK PIPELINE")
        print("EXOPLANET DETECTION SYSTEM")
        print("="*70)
        
        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
        
        # Build model
        input_dim = X_train.shape[1]
        self.build_model(input_dim, architecture=architecture)
        
        # Print model summary
        self.print_model_summary()
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val, 
                        epochs=epochs, batch_size=batch_size)
        
        # Evaluate model
        y_pred, y_pred_prob = self.evaluate_model(X_test, y_test)
        
        # Generate visualizations
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_prob)
        
        # Save model and preprocessors
        self.save_model_and_preprocessors()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - best_exoplanet_model.h5 (checkpoint model)")
        print("  - saved_model/ directory (all model components)")
        print("  - training_history.png")
        print("  - confusion_matrix.png")
        print("  - roc_curve.png")
        
        return self.model, self.history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create deep neural network instance
    dnn = ExoplanetDeepNeuralNetwork(
        train_path='data/train.csv',
        val_path='data/validation.csv',
        test_path='data/test.csv'
    )
    
    # Run complete pipeline
    model, history = dnn.run_complete_pipeline(
        epochs=100,
        batch_size=32,
        architecture='deep'  # Options: 'deep' or 'wide'
    )
    
    print("\n🚀 Deep learning model training complete!")
    print("✅ Model and all components exported successfully!")