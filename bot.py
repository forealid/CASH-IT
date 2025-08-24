#!/usr/bin/env python3
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List
import asyncio

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Update
from dotenv import load_dotenv

from database import DatabaseManager, CrashHistory, BotState, LiveStats, HourlyStats
from ml import MLEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
TIMEZONE = ZoneInfo("Africa/Algiers")
MAX_CAP = 4.5
CATEGORIES = {
    'low': lambda x: x < 1.5,
    'medium': lambda x: 1.5 <= x < 2.0,
    'high': lambda x: x >= 2.0
}

class CrashBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN environment variable is required")
        
        # Initialize components
        self.db = DatabaseManager()
        self.ml_engine = MLEngine(self.db)
        
        # Admin configuration
        admins_str = os.getenv('ADMINS', '')
        self.admins = set(map(int, admins_str.split())) if admins_str else set()
        
        # Initialize updater
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all command and message handlers"""
        self.dispatcher.add_handler(CommandHandler("start", self.start))
        self.dispatcher.add_handler(CommandHandler("train", self.train_mode))
        self.dispatcher.add_handler(CommandHandler("play", self.play_mode))
        self.dispatcher.add_handler(CommandHandler("stats", self.show_stats))
        self.dispatcher.add_handler(CommandHandler("times", self.show_times))
        self.dispatcher.add_handler(CommandHandler("backtest", self.backtest))
        self.dispatcher.add_handler(CommandHandler("trainmodel", self.train_model))
        self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.handle_message))
    
    def _is_admin(self, user_id: int) -> bool:
        """Check if user is admin"""
        return not self.admins or user_id in self.admins
    
    def _get_category(self, value: float) -> str:
        """Get category for crash value"""
        if CATEGORIES['low'](value):
            return 'low'
        elif CATEGORIES['medium'](value):
            return 'medium'
        else:
            return 'high'
    
    def _clamp_value(self, value: float) -> float:
        """Clamp value to valid range"""
        return max(1.01, min(MAX_CAP, value))
    
    async def start(self, update: Update, context: CallbackContext):
        """Start command handler"""
        welcome_msg = """🎯 **Crash Prediction Oracle**

Available commands:
/train - Switch to training mode (log only)
/play - Switch to play mode (log + predict)
/stats - Show live performance stats
/times - Show best day/hour analysis
/backtest N - Run backtest on last N rounds

Send crash values (1.01-4.50) to log data."""
        
        await update.message.reply_text(welcome_msg)
    
    async def train_mode(self, update: Update, context: CallbackContext):
        """Switch to training mode"""
        chat_id = update.effective_chat.id
        self.db.set_bot_mode(chat_id, 'train')
        
        await update.message.reply_text(
            "🔄 **Training Mode Activated**\n\n"
            "Send crash values to build the prediction model.\n"
            "Values will be logged without predictions."
        )
    
    async def play_mode(self, update: Update, context: CallbackContext):
        """Switch to play mode"""
        chat_id = update.effective_chat.id
        self.db.set_bot_mode(chat_id, 'play')
        
        await update.message.reply_text(
            "🎮 **Play Mode Activated**\n\n"
            "Send crash values to log data and receive predictions.\n"
            "Performance will be tracked automatically."
        )
    
    async def show_stats(self, update: Update, context: CallbackContext):
        """Show live statistics"""
        stats = self.db.get_live_stats()
        
        if stats.total == 0:
            await update.message.reply_text(
                "📊 **Live Performance**\n\n"
                "No predictions made yet.\n"
                "Use /play mode and submit values to start tracking."
            )
            return
        
        win_rate = (stats.wins / stats.total * 100) if stats.total > 0 else 0
        
        stats_msg = f"""📊 **Live Performance**

Since: {stats.since}
Total Predictions: {stats.total}
Wins: {stats.wins}
Losses: {stats.losses}
Win Rate: {win_rate:.1f}%"""
        
        await update.message.reply_text(stats_msg)
    
    async def show_times(self, update: Update, context: CallbackContext):
        """Show best day/hour analysis"""
        times_data = self.db.get_times_analysis()
        
        if not times_data:
            await update.message.reply_text(
                "⏰ **Best Times Analysis**\n\n"
                "Not enough data for analysis.\n"
                "Log more crash values to see patterns."
            )
            return
        
        # Format times table
        times_msg = "⏰ **Best Times Analysis**\n\n"
        times_msg += "Day | Hour | High% | Total\n"
        times_msg += "---|---|---|---\n"
        
        for row in times_data[:10]:  # Show top 10
            day, hour, high_pct, total = row
            times_msg += f"{day} | {hour:02d}:00 | {high_pct:.1f}% | {total}\n"
        
        await update.message.reply_text(f"```\n{times_msg}\n```", parse_mode='Markdown')
    
    async def backtest(self, update: Update, context: CallbackContext):
        """Run backtest on historical data"""
        try:
            # Parse number of rounds
            if not context.args:
                await update.message.reply_text("Usage: /backtest N (where N is number of rounds)")
                return
            
            n_rounds = int(context.args[0])
            if n_rounds <= 0:
                await update.message.reply_text("Please specify a positive number of rounds")
                return
            
            # Run backtest
            results = self.ml_engine.run_backtest(n_rounds)
            
            if not results:
                await update.message.reply_text("Not enough historical data for backtest")
                return
            
            backtest_msg = f"""📈 **Backtest Results ({n_rounds} rounds)**

Win Rate: {results['win_rate']:.1f}%
MAE: {results['mae']:.3f}
Profit: {results['profit']:+.2f} units
Predictions: {results['total']} total"""
            
            await update.message.reply_text(backtest_msg)
            
        except ValueError:
            await update.message.reply_text("Please provide a valid number")
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            await update.message.reply_text("Error running backtest")
    
    async def train_model(self, update: Update, context: CallbackContext):
        """Train ML model (admin only)"""
        user_id = update.effective_user.id
        
        if not self._is_admin(user_id):
            await update.message.reply_text("⛔ Admin access required")
            return
        
        await update.message.reply_text("🔄 **Training model...**\n\nThis may take a moment.")
        
        try:
            # Train model
            success, message = self.ml_engine.train_model()
            
            if success:
                # Reset live stats
                self.db.reset_live_stats()
                await update.message.reply_text(f"✅ **Model trained successfully**\n\n{message}")
            else:
                await update.message.reply_text(f"❌ **Training failed**\n\n{message}")
                
        except Exception as e:
            logger.error(f"Model training error: {e}")
            await update.message.reply_text("❌ **Training failed**\n\nUnexpected error occurred")
    
    async def handle_message(self, update: Update, context: CallbackContext):
        """Handle crash value inputs"""
        try:
            # Parse crash value
            text = update.message.text.strip()
            value = float(text.replace('x', ''))
            
            # Validate and clamp value
            if value < 1.01:
                await update.message.reply_text("⚠️ Value must be at least 1.01")
                return
            
            value = self._clamp_value(value)
            chat_id = update.effective_chat.id
            
            # Get current time
            now = datetime.now(TIMEZONE)
            time_str = now.strftime("%H:%M")
            hour = now.hour
            day_name = now.strftime("%A")
            
            # Determine category
            category = self._get_category(value)
            
            # Get bot state
            bot_state = self.db.get_bot_state(chat_id)
            mode = bot_state.mode if bot_state else 'train'
            
            # Log the crash value
            self.db.add_crash_value(value, category, time_str, hour, day_name)
            self.db.update_hourly_stats(day_name, hour, category)
            
            if mode == 'train':
                # Training mode - just log
                response = f"""📝 **Logged (Training Mode)**

Value: {value:.2f}x
Category: {category.title()}
Time: {time_str}
Day: {day_name}

Use /play to switch to prediction mode."""
                
            else:
                # Play mode - score previous and predict next
                response_parts = []
                
                # Score previous prediction if exists
                if bot_state and bot_state.last_prediction:
                    predicted = bot_state.last_prediction
                    actual = value
                    
                    # Determine if prediction was correct (within category)
                    pred_category = self._get_category(predicted)
                    actual_category = self._get_category(actual)
                    
                    is_win = pred_category == actual_category
                    
                    # Update live stats
                    self.db.update_live_stats(is_win)
                    
                    # Format previous result
                    result_emoji = "✅" if is_win else "❌"
                    response_parts.append(f"""🎯 **Previous Result**

{result_emoji} Predicted: {predicted:.2f}x ({pred_category})
Actual: {actual:.2f}x ({actual_category})""")
                
                # Log current value
                response_parts.append(f"""📝 **Current Log**

Value: {value:.2f}x
Category: {category.title()}
Time: {time_str}
Day: {day_name}""")
                
                # Make new prediction
                prediction, confidence = self.ml_engine.predict_next()
                prediction = round(prediction, 2)
                confidence_pct = round(confidence * 100, 1)
                pred_category = self._get_category(prediction)
                
                # Save prediction
                self.db.set_last_prediction(chat_id, prediction, time_str)
                
                response_parts.append(f"""🔮 **Next Prediction**

Predicted: {prediction:.2f}x ({pred_category})
Confidence: {confidence_pct}%""")
                
                response = "\n\n".join(response_parts)
            
            await update.message.reply_text(response)
            
        except ValueError:
            await update.message.reply_text(
                "⚠️ Please send a valid crash value (e.g., 1.25, 2.50x, 3.75)"
            )
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await update.message.reply_text("❌ Error processing value")
    
    def start_bot(self):
        """Start the bot"""
        logger.info("Starting Crash Prediction Oracle...")
        
        # Initialize database
        self.db.init_database()
        
        # Load ML model if exists
        self.ml_engine.load_model()
        
        # Start polling
        self.updater.start_polling()
        logger.info("Bot is running...")
        
        # Keep running
        self.updater.idle()

def main():
    try:
        bot = CrashBot()
        bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise

if __name__ == '__main__':
    main()
