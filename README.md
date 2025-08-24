# Crash Prediction Oracle Bot

A production-ready Telegram bot that predicts crash values using XGBoost machine learning with 24/7 uptime on Railway.

## Features

- **Dual Modes**: Training mode (log only) and Play mode (log + predict)
- **ML Prediction**: XGBoost regression with engineered features
- **Performance Tracking**: Live win/loss statistics and backtesting
- **Time Analysis**: Best day/hour patterns for high values
- **PostgreSQL Storage**: Persistent data with optimized indexes
- **Admin Controls**: Restricted model training access

## Commands

- `/train` - Switch to training mode (log values only)
- `/play` - Switch to play mode (log + predict + score)
- `/stats` - Show live performance statistics
- `/times` - Show best day/hour analysis for high values
- `/backtest N` - Run backtest on last N rounds
- `/trainmodel` - Train ML model (admin only)

## Quick Start

### A) Local Development

1. **Clone and setup environment:**
```bash
git clone <your-repo>
cd crash-prediction-bot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your values:
# - Get TELEGRAM_TOKEN from @BotFather on Telegram
# - Set DATABASE_URL to your PostgreSQL instance
# - Add your Telegram user ID to ADMINS
```

3. **Run locally:**
```bash
python bot.py
```

4. **Test the bot:**
- Send `/train` to start logging values
- Send crash values like `1.25`, `2.50x`, `3.75`
- Use `/trainmodel` to train ML model (admin only)
- Switch to `/play` mode for predictions
- Check `/stats`, `/backtest 100`, and `/times`

### B) Deploy to Railway (24/7 Hosting)

1. **Push to GitHub:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app) and sign up
   - Create new project → Deploy from GitHub repo
   - Add PostgreSQL database add-on to your project

3. **Configure environment variables in Railway dashboard:**
```
TELEGRAM_TOKEN=your_bot_token_from_botfather
DATABASE_URL=postgresql://... (auto-provided by Railway Postgres)
ADMINS=your_telegram_user_id
MODEL_PATH=/app/model.json
MAX_TRAIN_ROWS=200000
```

4. **Deploy automatically:**
   - Railway detects `Procfile` and deploys as worker process
   - Check logs in Railway dashboard
   - Bot runs 24/7 with automatic restarts

### C) Usage Workflow

1. **Data Collection Phase:**
   - Use `/train` mode to log crash values: `1.25`, `2.30x`, `4.10`
   - Bot stores values with timestamps and categories
   - Collect 100+ values before training

2. **Model Training:**
   - Admin runs `/trainmodel` to train XGBoost model
   - Features include streaks, volatility, time patterns
   - Model saved automatically for future predictions

3. **Prediction Phase:**
   - Switch to `/play` mode
   - Bot logs values AND makes predictions for next round
   - Automatically scores previous predictions (win/loss)
   - Live stats updated in real-time

4. **Analysis & Monitoring:**
   - `/stats` - Check win rate and performance
   - `/backtest 1000` - Validate model on historical data
   - `/times` - Find best hours/days for high values (≥2.0x)

## Technical Details

### Database Schema

- **crash_history**: All logged values with timestamps
- **hourly_stats**: Aggregated counts by day/hour for analysis
- **bot_state**: Per-chat mode and last prediction tracking  
- **live_stats**: Real-time performance metrics

### ML Features

- Consecutive low/medium streaks
- Rounds since last high value
- Rolling statistics (median, std, max, min)
- Category distribution percentages
- Time-based features (hour, day)
- Volatility and trend measures

### Categories

- **Low**: < 1.5x
- **Medium**: 1.5x to < 2.0x  
- **High**: 2.0x to 4.5x (capped)

### Performance

- Supports thousands of records with PostgreSQL indexes
- Configurable training data limit via `MAX_TRAIN_ROWS`
- Automatic model persistence and loading
- Fallback heuristic predictions if ML model unavailable

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_TOKEN` | Bot token from @BotFather | Required |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `ADMINS` | Space-separated Telegram user IDs | None (all users can train) |
| `MODEL_PATH` | Path to save/load ML model | `model.json` |
| `MAX_TRAIN_ROWS` | Maximum rows for model training | `200000` |

## Troubleshooting

**Bot not responding:** Check Railway logs for errors, verify `TELEGRAM_TOKEN`

**Database errors:** Ensure `DATABASE_URL` is correct and PostgreSQL is running

**Prediction failures:** Check if model exists, try `/trainmodel` with more data

**Admin commands blocked:** Verify your Telegram user ID is in `ADMINS`

## Support

For Railway-specific deployment issues, check:
- [Railway Documentation](https://docs.railway.app)
- Railway dashboard logs and metrics
- PostgreSQL add-on connection details
