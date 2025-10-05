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
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class ExoplanetDeepNeuralNetwork:
    """
    Enhanced Deep Neural Network for Exoplanet Detection.
    Features: Outlier removal, LeakyReLU, L2 regularization, cosine decay with warm restarts.
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
        
    def detect_and_remove_outliers(self, X, y, method='iqr', threshold=3.0):
        """
        Detect and remove outliers using IQR or Z-score method.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        method : str
            'iqr' or 'zscore'
        threshold : float
            For IQR: multiplier (default 3.0)
            For Z-score: number of std deviations (default 3.0)
            
        Returns:
        --------
        X_clean, y_clean : cleaned data without outliers
        """
        print(f"\nDetecting outliers using {method.upper()} method...")
        print(f"Original data shape: {X.shape}")
        
        if method == 'iqr':
            # IQR method
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            outlier_mask = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((X - X.mean()) / X.std())
            outlier_mask = (z_scores > threshold).any(axis=1)
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        # Remove outliers
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        outliers_removed = outlier_mask.sum()
        print(f"Outliers detected: {outliers_removed} ({outliers_removed/len(X)*100:.2f}%)")
        print(f"Clean data shape: {X_clean.shape}")
        
        return X_clean, y_clean
        
    def load_and_preprocess_data(self, remove_outliers=True, outlier_method='iqr'):
        """
        Load and preprocess the train, validation, and test datasets.
        
        Parameters:
        -----------
        remove_outliers : bool
            Whether to remove outliers from training data
        outlier_method : str
            Method for outlier detection ('iqr' or 'zscore')
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
        
        # Remove outliers from training data only
        if remove_outliers:
            print("\n" + "="*70)
            print("OUTLIER DETECTION AND REMOVAL")
            print("="*70)
            X_train, y_train = self.detect_and_remove_outliers(
                X_train, y_train, method=outlier_method, threshold=3.0
            )
            print("\nClass distribution after outlier removal:")
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
    
    def build_model(self, input_dim, architecture='deep', l2_lambda=0.001):
        """
        Build a deep neural network model with LeakyReLU and L2 regularization.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        architecture : str
            Model architecture type: 'deep', 'wide', or 'custom'
        l2_lambda : float
            L2 regularization strength
        """
        print("\n" + "="*70)
        print("BUILDING ENHANCED DEEP NEURAL NETWORK")
        print("="*70)
        
        model = models.Sequential(name='Exoplanet_DNN_Enhanced')
        
        if architecture == 'deep':
            # Deep architecture with LeakyReLU and L2 regularization
            print("\nArchitecture: Deep Network (6 hidden layers)")
            print(f"Activation: LeakyReLU (alpha=0.2)")
            print(f"Regularization: L2 (lambda={l2_lambda})")
            print("-" * 70)
            
            # Input layer
            model.add(layers.Input(shape=(input_dim,), name='input_layer'))
            
            # Hidden Layer 1
            model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_1'))
            model.add(layers.BatchNormalization(name='bn_1'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_1'))
            model.add(layers.Dropout(0.3, name='dropout_1'))
            
            # Hidden Layer 2
            model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_2'))
            model.add(layers.BatchNormalization(name='bn_2'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_2'))
            model.add(layers.Dropout(0.3, name='dropout_2'))
            
            # Hidden Layer 3
            model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_3'))
            model.add(layers.BatchNormalization(name='bn_3'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_3'))
            model.add(layers.Dropout(0.3, name='dropout_3'))
            
            # Hidden Layer 4
            model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_4'))
            model.add(layers.BatchNormalization(name='bn_4'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_4'))
            model.add(layers.Dropout(0.2, name='dropout_4'))
            
            # Hidden Layer 5
            model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_5'))
            model.add(layers.BatchNormalization(name='bn_5'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_5'))
            model.add(layers.Dropout(0.2, name='dropout_5'))
            
            # Hidden Layer 6
            model.add(layers.Dense(8, kernel_regularizer=regularizers.l2(l2_lambda), name='dense_6'))
            model.add(layers.LeakyReLU(alpha=0.2, name='leaky_relu_6'))
            
            # Output layer
            model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
            
            print("\nLayer Configuration:")
            print(f"  Input Layer: {input_dim} neurons")
            print(f"  Hidden Layer 1: 256 neurons (LeakyReLU + BatchNorm + Dropout 0.3 + L2)")
            print(f"  Hidden Layer 2: 128 neurons (LeakyReLU + BatchNorm + Dropout 0.3 + L2)")
            print(f"  Hidden Layer 3: 64 neurons (LeakyReLU + BatchNorm + Dropout 0.2 + L2)")
            print(f"  Hidden Layer 4: 32 neurons (LeakyReLU + BatchNorm + Dropout 0.2 + L2)")
            print(f"  Hidden Layer 5: 16 neurons (LeakyReLU + BatchNorm + Dropout 0.1 + L2)")
            print(f"  Hidden Layer 6: 8 neurons (LeakyReLU + L2)")
            print(f"  Output Layer: 1 neuron (Sigmoid)")
        
        elif architecture == 'wide':
            # Wide architecture with LeakyReLU and L2
            print("\nArchitecture: Wide Network (4 hidden layers)")
            print(f"Activation: LeakyReLU (alpha=0.2)")
            print(f"Regularization: L2 (lambda={l2_lambda})")
            print("-" * 70)
            
            model.add(layers.Input(shape=(input_dim,)))
            
            model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(l2_lambda)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.4))
            
            model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(l2_lambda)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.3))
            
            model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda)))
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.2))
            
            model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda)))
            model.add(layers.LeakyReLU(alpha=0.2))
            
            model.add(layers.Dense(1, activation='sigmoid'))
        
        # Learning rate schedule with warm restarts
        first_decay_steps = 10
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=0.0001,
            first_decay_steps=first_decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0
        )
        
        # Compile model with scheduled learning rate
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\n" + "="*70)
        print("MODEL COMPILATION COMPLETED")
        print("="*70)
        print(f"Optimizer: Adam with Cosine Decay Restarts")
        print(f"  - Initial LR: 0.001")
        print(f"  - First Decay Steps: {first_decay_steps}")
        print(f"  - T_mul: 2.0 (doubles restart period)")
        print(f"  - M_mul: 0.9 (reduces LR by 10% each restart)")
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
        print("TRAINING ENHANCED DEEP NEURAL NETWORK")
        print("="*70)
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Training Samples: {len(X_train)}")
        print(f"  Validation Samples: {len(X_val)}")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'best_exoplanet_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        print("\nCallbacks configured:")
        print("  - Early Stopping (patience=20)")
        print("  - Model Checkpoint (saving best model)")
        print("  - Cosine Decay with Warm Restarts (automatic)")
        
        print("\n" + "="*70)
        print("Starting Training...")
        print("="*70 + "\n")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
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
            'model_name': 'Enhanced Exoplanet Deep Neural Network',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist(),
            'architecture': 'deep',
            'activation': 'LeakyReLU',
            'regularization': 'L2',
            'lr_schedule': 'CosineDecayRestarts',
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
        
        plt.suptitle('Enhanced Deep Neural Network Training History', 
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
        
        plt.title('Confusion Matrix - Enhanced Deep Neural Network\nExoplanet Detection', 
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
        plt.title('ROC Curve - Enhanced Deep Neural Network\nExoplanet Detection', 
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
    
    def run_complete_pipeline(self, epochs=100, batch_size=32, architecture='deep',
                            remove_outliers=True, outlier_method='iqr', l2_lambda=0.001):
        """
        Run the complete enhanced deep learning pipeline.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        architecture : str
            'deep' or 'wide'
        remove_outliers : bool
            Whether to remove outliers
        outlier_method : str
            'iqr' or 'zscore'
        l2_lambda : float
            L2 regularization strength
        """
        print("\n" + "="*70)
        print("STARTING ENHANCED DEEP NEURAL NETWORK PIPELINE")
        print("EXOPLANET DETECTION SYSTEM")
        print("="*70)
        print("\nEnhancements:")
        print("  ✓ Outlier Detection and Removal")
        print("  ✓ LeakyReLU Activation (alpha=0.2)")
        print("  ✓ L2 Regularization on Dense Layers")
        print("  ✓ Cosine Decay with Warm Restarts Learning Rate Schedule")
        print("="*70)
        
        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data(
            remove_outliers=remove_outliers,
            outlier_method=outlier_method
        )
        
        # Build model
        input_dim = X_train.shape[1]
        self.build_model(input_dim, architecture=architecture, l2_lambda=l2_lambda)
        
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
    
    # Create enhanced deep neural network instance
    dnn = ExoplanetDeepNeuralNetwork(
        train_path='data/train.csv',
        val_path='data/validation.csv',
        test_path='data/test.csv'
    )
    
    # Run complete pipeline with enhancements
    model, history = dnn.run_complete_pipeline(
        epochs=100,
        batch_size=32,
        architecture='deep',        # Options: 'deep' or 'wide'
        remove_outliers=True,       # Enable outlier removal
        outlier_method='iqr',       # Options: 'iqr' or 'zscore'
        l2_lambda=0.001             # L2 regularization strength
    )
    
    print("\n🚀 Enhanced deep learning model training complete!")
    print("✅ Model with all enhancements exported successfully!")
    print("\n📊 Key Features:")
    print("   • Outlier detection and removal")
    print("   • LeakyReLU activation functions")
    print("   • L2 regularization on all dense layers")
    print("   • Cosine decay with warm restarts learning rate schedule")