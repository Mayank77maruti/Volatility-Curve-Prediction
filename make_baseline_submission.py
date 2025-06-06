import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization, 
                                    Attention, Concatenate, LayerNormalization)
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import backend as K
# from tensorflow.keras.layers import layer

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import os
import time

class AdvancedVolatilityPredictor:
    def __init__(self, sequence_length=30, lstm_units=128):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.scaler = None
        self.model = None
        self.feature_columns = []
        self.target_columns = []
        self.time_scaler = None
        self.min_timestamp = None
        
    def load_data(self, train_path, test_path, sample_sub_path):
        print("Loading datasets...")
        self.train_df = pd.read_parquet(train_path)
        self.test_df = pd.read_parquet(test_path)
        self.sample_sub = pd.read_csv(sample_sub_path)
        
        # Convert timestamp columns to datetime
        self.train_df['timestamp'] = pd.to_datetime(self.train_df['timestamp'])
        self.test_df['timestamp'] = pd.to_datetime(self.test_df['timestamp'])

        # Store min timestamp for normalization
        self.min_timestamp = min(self.train_df['timestamp'].min(), self.test_df['timestamp'].min())
        
        print(f"Train shape: {self.train_df.shape}, Test shape: {self.test_df.shape}")
        
    def engineer_features(self, df):
        """Create advanced features for volatility prediction"""
        df = df.copy()
        
        # Convert timestamp to numerical features
        df['seconds'] = (df['timestamp'] - self.min_timestamp).dt.total_seconds()
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['time_of_day'] = df['hour'] + df['minute']/60
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
        
        # Price dynamics
        if 'underlying_price' in df.columns:
            # Logarithmic returns
            df['log_return'] = np.log(df['underlying_price'] / df['underlying_price'].shift(1))
            
            # Volatility metrics
            for window in [5, 15, 30, 60]:  # 5s, 15s, 30s, 1m windows
                df[f'volatility_{window}s'] = df['log_return'].rolling(window).std()
                df[f'return_ma_{window}s'] = df['underlying_price'].rolling(window).mean()
            
            # Price momentum
            df['price_change_1s'] = df['underlying_price'].diff()
            df['price_accel'] = df['price_change_1s'].diff()
            
            # Technical indicators
            df['high_low_spread'] = df['high'] - df['low'] if 'high' in df.columns else 0
            df['close_open_spread'] = df['close'] - df['open'] if 'close' in df.columns else 0
        
        # Volume features (if available)
        volume_cols = [c for c in df.columns if 'volume' in c.lower()]
        for col in volume_cols:
            df[f'{col}_change'] = df[col].diff()
            for window in [5, 15]:
                df[f'{col}_rolling_mean_{window}s'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}s'] = df[col].rolling(window).std()
        
        # ATM IV features
        if 'atm_iv' in df.columns:
            df['atm_iv_change'] = df['atm_iv'].diff()
            df['atm_iv_pct_change'] = df['atm_iv'].pct_change()
            
            # Volatility smile features
            for strike in [24000, 25000, 26000]:
                if f'call_iv_{strike}' in df.columns and f'put_iv_{strike}' in df.columns:
                    df[f'skew_{strike}'] = df[f'call_iv_{strike}'] - df[f'put_iv_{strike}']
        
        # Fill NaNs using forward then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Drop infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def prepare_data(self):
        """Prepare data for LSTM training"""
        # Identify target IV columns
        self.target_columns = [col for col in self.train_df.columns 
                              if col.startswith(('call_iv_', 'put_iv_'))]
        
        print(f"Found {len(self.target_columns)} target IV columns")
        
        # Prepare features (exclude targets and timestamp)
        feature_cols = [col for col in self.train_df.columns 
                       if col not in self.target_columns + ['timestamp']]
        
        # Select only numeric features
        X_train = self.train_df[feature_cols].select_dtypes(include=np.number)
        self.feature_columns = X_train.columns.tolist()
        y_train = self.train_df[self.target_columns]
        
        print(f"Using {len(self.feature_columns)} features for modeling")
        
        # Feature scaling - use RobustScaler to handle outliers
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences for LSTM
        X_sequences, y_sequences = [], []
        for i in range(len(X_train_scaled) - self.sequence_length):
            X_sequences.append(X_train_scaled[i:i+self.sequence_length])
            y_sequences.append(y_train.iloc[i+self.sequence_length].values)
            
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Sequences shape: {X_sequences.shape}, Targets shape: {y_sequences.shape}")
        return X_sequences, y_sequences
    
    def hybrid_loss(self, y_true, y_pred):
        """Hybrid loss combining likelihood and MSE"""
        # Likelihood component
        epsilon = 1e-7
        iv_safe = tf.maximum(y_pred, epsilon)
        
        # Assuming first feature is log_return
        returns = y_true[:, 0]  
        log_term = 2 * tf.math.log(iv_safe)
        return_term = tf.square(returns[:, tf.newaxis]) / tf.square(iv_safe)
        likelihood_loss = tf.reduce_mean(log_term + return_term)
        
        # MSE component
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        
        # Combine (weighted sum)
        return 0.7 * likelihood_loss + 0.3 * mse_loss
    
    def build_model(self, input_shape, output_dim):
        """Build LSTM model with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # First LSTM layer (returns sequences for attention)
        x = LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second LSTM layer
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        context_vector = Attention()([x, x])
        context_vector = LayerNormalization()(context_vector)
        
        # Third LSTM layer (processes context vector)
        x = LSTM(self.lstm_units)(context_vector)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(output_dim, activation='softplus')(x)  # Ensures positive volatility
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss=self.hybrid_loss
        )
        
        return model
    
    def train_model(self, X, y):
        """Train the LSTM model with time-series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        best_val_loss = float('inf')
        best_model = None
        
        print("\nStarting time-series cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nTraining fold {fold+1}/{tscv.n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model
            model = self.build_model(
                input_shape=(self.sequence_length, len(self.feature_columns)),
                output_dim=len(self.target_columns)
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=256,
                callbacks=callbacks,
                verbose=1
            )
            
            # Track best model
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                print(f"New best model with val_loss: {best_val_loss:.4f}")
        
        self.model = best_model
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        
    def create_sequences_test(self, df):
        """Create sequences for test data"""
        # Prepare features
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X_test = df[self.feature_columns].select_dtypes(include=np.number)
        X_test = self.scaler.transform(X_test)
        
        # Create sequences with padding if needed
        X_sequences = []
        start_idx = max(0, len(X_test) - self.sequence_length)
        sequence = X_test[start_idx:]
        
        # Pad if sequence is shorter than required
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), X_test.shape[1]))
            sequence = np.vstack([padding, sequence])
        
        X_sequences.append(sequence)
        return np.array(X_sequences)
    
    def predict(self, test_df):
        """Generate predictions for test data"""
        # Create sequences for test data
        X_test_seq = self.create_sequences_test(test_df)
        
        # Generate predictions
        preds = self.model.predict(X_test_seq)
        
        # Create DataFrame with predictions
        predictions = pd.DataFrame(preds, columns=self.target_columns)
        
        # Add timestamp (use last timestamp in test data)
        predictions['timestamp'] = test_df['timestamp'].iloc[-1]
        
        # Clip predictions to reasonable range
        predictions[self.target_columns] = predictions[self.target_columns].clip(0.05, 1.5)
        
        return predictions
    
    def create_submission(self, predictions):
        """Create submission file in required format"""
        # Create full submission dataframe with all timestamps
        submission_df = self.sample_sub.copy()
        
        # Merge predictions with submission template
        for col in self.target_columns:
            if col in predictions.columns:
                # Fill all rows with the prediction (single timestamp)
                submission_df[col] = predictions[col].values[0]
        
        # Handle missing columns
        submission_cols = submission_df.columns.tolist()
        iv_cols = [col for col in submission_cols if col != 'timestamp']
        
        for col in iv_cols:
            if col not in predictions.columns:
                # For missing columns, use strike mapping to find similar
                try:
                    if col.startswith('call_iv_'):
                        strike = int(col.split('_')[-1])
                        available = [c for c in predictions.columns if c.startswith('call_iv_')]
                        strikes = [int(c.split('_')[-1]) for c in available]
                        closest_strike = min(strikes, key=lambda x: abs(x - strike))
                        submission_df[col] = predictions[f'call_iv_{closest_strike}'].values[0]
                    elif col.startswith('put_iv_'):
                        strike = int(col.split('_')[-1])
                        available = [c for c in predictions.columns if c.startswith('put_iv_')]
                        strikes = [int(c.split('_')[-1]) for c in available]
                        closest_strike = min(strikes, key=lambda x: abs(x - strike))
                        submission_df[col] = predictions[f'put_iv_{closest_strike}'].values[0]
                except:
                    submission_df[col] = 0.2
        
        # Final checks
        print(f"Submission shape: {submission_df.shape}")
        print(f"NaN values: {submission_df.isna().sum().sum()}")
        
        # Save submission
        output_path = os.path.join(os.getcwd(), "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
        return submission_df

def main():
    """Main execution function"""
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, "train_data.parquet")
    test_path = os.path.join(current_dir, "test_data.parquet")
    sample_sub_path = os.path.join(current_dir, "sample_submission.csv")
    
    # Initialize and run predictor
    predictor = AdvancedVolatilityPredictor(sequence_length=30, lstm_units=128)
    predictor.load_data(train_path, test_path, sample_sub_path)
    
    # Feature engineering
    print("\nEngineering features for training data...")
    start_time = time.time()
    predictor.train_df = predictor.engineer_features(predictor.train_df)
    print(f"Feature engineering completed in {time.time()-start_time:.2f} seconds")
    
    # Prepare training data
    print("\nPreparing training sequences...")
    start_time = time.time()
    X, y = predictor.prepare_data()
    print(f"Data preparation completed in {time.time()-start_time:.2f} seconds")
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    predictor.train_model(X, y)
    print(f"Model training completed in {time.time()-start_time:.2f} seconds")
    
    # Prepare test data
    print("\nEngineering features for test data...")
    predictor.test_df = predictor.engineer_features(predictor.test_df)
    
    # Create predictions
    print("\nCreating predictions...")
    start_time = time.time()
    predictions = predictor.predict(predictor.test_df)
    print(f"Prediction completed in {time.time()-start_time:.2f} seconds")
    
    # Create submission
    print("\nCreating submission...")
    submission = predictor.create_submission(predictions)
    
    print("\nSubmission preview:")
    print(submission.head())

if __name__ == "__main__":
    # Configure TensorFlow for better performance
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.set_soft_device_placement(True)
    
    # Set memory growth to prevent GPU OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()