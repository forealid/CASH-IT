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
        
        if len
