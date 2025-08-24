#!/usr/bin/env python3
import os
import json
import logging
from typing import Tuple, List, Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

class MLEngine:
    def __init__(self, db_manager):
        self.db = db_manager
        self.model = None
        self.model_path = os.getenv('MODEL_PATH', 'model.json')
        self.max_train_rows = int(os.getenv('MAX_TRAIN_ROWS', '200000'))
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Categories
        self.categories = {
            'low': lambda x: x < 1.5,
            'medium': lambda x: 1.5 <= x < 2.0,
            'high': lambda x: x >= 2.0
        }
        
        self.max_cap = 4.5
    
    def load_model(self) -> bool:
        """Load trained model from file"""
        try:
            if os.path.exists(self.model_path):
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_path)
                logger.info("ML model loaded successfully")
                return True
            else:
                logger.info("No existing model found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self) -> bool:
        """Save trained model to file"""
        try:
            if self.model:
                self.model.save_model(self.model_path)
                logger.info("ML model saved successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _get_category(self, value: float) -> str:
        """Get category for crash value"""
        if self.categories['low'](value):
            return 'low'
        elif self.categories['medium'](value):
            return 'medium'
        else:
            return 'high'
    
    def _clamp_value(self, value: float) -> float:
        """Clamp value to valid range"""
        return max(1.01, min(self.max_cap, value))
    
    def _extract_features(self, crash_history: List, index: int) -> List[float]:
        """Extract features for ML model at given index"""
        if index < 50:  # Need at least 50 previous values
            return None
        
        # Get recent values before current index
        recent_values = [crash_history[i].crash_value for i in range(max(0, index-50), index)]
        
        if len(recent_values) < 10:
            return None
        
        features = []
        
        # 1. Consecutive streaks
        low_streak = 0
        med_streak = 0
        for i in range(len(recent_values) - 1, -1, -1):
            val = recent_values[i]
            if val < 1.5:  # low
                if med_streak == 0:
                    low_streak += 1
                else:
                    break
            elif 1.5 <= val < 2.0:  # medium
                if low_streak == 0:
                    med_streak += 1
                else:
                    break
            else:
                break
        
        features.extend([low_streak, med_streak])
        
        # 2. Rounds since last high
        rounds_since_high = 0
        for i in range(len(recent_values) - 1, -1, -1):
            if recent_values[i] >= 2.0:
                break
            rounds_since_high += 1
        
        features.append(rounds_since_high)
        
        # 3. Rolling statistics (last 50)
        recent_array = np.array(recent_values)
        features.extend([
            np.median(recent_array),
            np.std(recent_array),
            np.max(recent_array),
            np.mean(recent_array),
            np.min(recent_array)
        ])
        
        # 4. Category percentages in recent history
        low_count = sum(1 for v in recent_values if v < 1.5)
        med_count = sum(1 for v in recent_values if 1.5 <= v < 2.0)
        high_count = sum(1 for v in recent_values if v >= 2.0)
        total = len(recent_values)
        
        features.extend([
            low_count / total,
            med_count / total,
            high_count / total
        ])
        
        # 5. Time-based features
        current_record = crash_history[index-1]  # Previous record for context
        features.extend([
            current_record.hour / 23.0,  # Normalize hour
            hash(current_record.day_name) % 7 / 6.0  # Day of week
        ])
        
        # 6. Volatility measures
        if len(recent_values) >= 10:
            last_10 = recent_values[-10:]
            volatility = np.std(last_10)
            features.append(volatility)
            
            # Trend (simple linear regression slope)
            x = np.arange(len(last_10))
            y = np.array(last_10)
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                features.append(slope)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def train_model(self) -> Tuple[bool, str]:
        """Train the ML model on crash history"""
        try:
            # Get crash history
            crash_history = self.db.get_crash_history(limit=self.max_train_rows)
            
            if len(crash_history) < 100:
                return False, "Not enough data (minimum 100 records required)"
            
            logger.info(f"Training with {len(crash_history)} records")
            
            # Prepare training data
            X, y = [], []
            
            for i in range(50, len(crash_history)):
                features = self._extract_features(crash_history, i)
                if features is not None:
                    X.append(features)
                    y.append(crash_history[i].crash_value)
            
            if len(X) < 50:
                return False, "Not enough valid feature vectors"
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train XGBoost model
            self.model = xgb.XGBRegressor(**self.xgb_params)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Save model
            self.save_model()
            
            message = f"Model trained successfully!\n"
            message += f"Training samples: {len(X_train)}\n"
            message += f"Test samples: {len(X_test)}\n"
            message += f"Test MAE: {mae:.3f}"
            
            return True, message
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False, f"Training error: {str(e)}"
    
    def predict_next(self) -> Tuple[float, float]:
        """Predict next crash value with confidence"""
        try:
            # Get recent crash history
            crash_history = self.db.get_crash_history(limit=100)
            
            if len(crash_history) < 50:
                return self._fallback_prediction()
            
            # Use ML model if available
            if self.model:
                features = self._extract_features(crash_history, len(crash_history))
                if features is not None:
                    features_array = np.array([features])
                    prediction = self.model.predict(features_array)[0]
                    prediction = self._clamp_value(prediction)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(crash_history, is_ml=True)
                    
                    return prediction, confidence
            
            # Fallback to heuristic prediction
            return self._fallback_prediction()
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> Tuple[float, float]:
        """Fallback heuristic prediction when ML model unavailable"""
        try:
            # Get recent values
            recent_values = self.db.get_recent_crash_values(limit=50)
            
            if len(recent_values) < 10:
                # Default prediction if no data
                return 1.85, 0.50
            
            # Simple heuristics based on recent patterns
            recent_array = np.array(recent_values)
            
            # Base prediction on rolling average with volatility adjustment
            rolling_mean = np.mean(recent_array[-20:]) if len(recent_array) >= 20 else np.mean(recent_array)
            volatility = np.std(recent_array[-10:]) if len(recent_array) >= 10 else np.std(recent_array)
            
            # Adjust based on recent trend
            if len(recent_array) >= 5:
                recent_trend = np.mean(recent_array[-5:]) - np.mean(recent_array[-10:-5]) if len(recent_array) >= 10 else 0
                rolling_mean += recent_trend * 0.3
            
            # Count recent categories
            low_count = sum(1 for v in recent_values[:20] if v < 1.5)
            high_count = sum(1 for v in recent_values[:20] if v >= 2.0)
            
            # Adjust prediction based on category distribution
            if low_count > high_count * 2:
                rolling_mean += 0.2  # Expect reversion to higher values
            elif high_count > low_count * 2:
                rolling_mean -= 0.15  # Expect reversion to lower values
            
            prediction = self._clamp_value(rolling_mean)
            confidence = self._calculate_confidence(None, is_ml=False)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            return 1.85, 0.50
    
    def _calculate_confidence(self, crash_history=None, is_ml=True) -> float:
        """Calculate prediction confidence"""
        base_confidence = 0.70 if is_ml else 0.60
        
        try:
            # Get recent data for confidence calculation
            recent_values = self.db.get_recent_crash_values(limit=50)
            
            if len(recent_values) < 10:
                return max(0.50, base_confidence - 0.15)
            
            # Adjust based on volatility
            volatility = np.std(recent_values)
            volatility_penalty = min(0.20, volatility * 0.1)
            
            # Adjust based on data volume
            data_count = len(self.db.get_crash_history(limit=1000))
            volume_bonus = min(0.15, data_count / 10000 * 0.15)
            
            confidence = base_confidence - volatility_penalty + volume_bonus
            
            # Clamp confidence
            return max(0.50, min(0.95, confidence))
            
        except Exception:
            return base_confidence
    
    def run_backtest(self, n_rounds: int) -> Optional[Dict]:
        """Run backtest on historical data"""
        try:
            # Get sufficient historical data
            crash_history = self.db.get_crash_history(limit=n_rounds + 100)
            
            if len(crash_history) < n_rounds + 50:
                return None
            
            # Reverse to chronological order
            crash_history = crash_history[::-1]
            
            predictions = []
            actuals = []
            wins = 0
            total = 0
            profit = 0.0
            
            # Start from index where we have enough history for features
            start_idx = max(50, len(crash_history) - n_rounds)
            
            for i in range(start_idx, len(crash_history)):
                # Make prediction based on history up to this point
                if self.model:
                    features = self._extract_features(crash_history, i)
                    if features is not None:
                        prediction = self.model.predict(np.array([features]))[0]
                        prediction = self._clamp_value(prediction)
                    else:
                        # Use recent average as fallback
                        recent = [crash_history[j].crash_value for j in range(max(0, i-10), i)]
                        prediction = np.mean(recent) if recent else 1.85
                else:
                    # Heuristic prediction
                    recent = [crash_history[j].crash_value for j in range(max(0, i-20), i)]
                    prediction = np.mean(recent) if recent else 1.85
                    prediction = self._clamp_value(prediction)
                
                actual = crash_history[i].crash_value
                
                predictions.append(prediction)
                actuals.append(actual)
                
                # Check if prediction category matches actual category
                pred_cat = self._get_category(prediction)
                actual_cat = self._get_category(actual)
                
                if pred_cat == actual_cat:
                    wins += 1
                    profit += 0.5  # Win profit
                else:
                    profit -= 1.0  # Loss
                
                total += 1
            
            if total == 0:
                return None
            
            # Calculate metrics
            win_rate = (wins / total) * 100
            mae = mean_absolute_error(actuals, predictions)
            
            return {
                'win_rate': win_rate,
                'mae': mae,
                'profit': profit,
                'total': total,
                'wins': wins,
                'losses': total - wins
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return None
