#!/usr/bin/env python3
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional
import logging

from sqlalchemy import create_engine, Column, Integer, Float, String, BigInteger, DateTime, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()

class CrashHistory(Base):
    __tablename__ = 'crash_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crash_value = Column(Float, nullable=False)
    category = Column(String(10), nullable=False)
    time = Column(String(5), nullable=False)
    hour = Column(Integer, nullable=False)
    day_name = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(ZoneInfo("Africa/Algiers")))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_crash_value', 'crash_value'),
        Index('idx_hour', 'hour'),
        Index('idx_day_name', 'day_name'),
        Index('idx_created_at', 'created_at'),
    )

class HourlyStats(Base):
    __tablename__ = 'hourly_stats'
    
    day_name = Column(String(10), primary_key=True)
    hour = Column(Integer, primary_key=True)
    low_count = Column(Integer, default=0)
    med_count = Column(Integer, default=0)
    high_count = Column(Integer, default=0)

class BotState(Base):
    __tablename__ = 'bot_state'
    
    chat_id = Column(BigInteger, primary_key=True)
    mode = Column(String(10), default='train')
    last_prediction = Column(Float, nullable=True)
    last_prediction_time = Column(String(5), nullable=True)

class LiveStats(Base):
    __tablename__ = 'live_stats'
    
    id = Column(Integer, primary_key=True, default=1)
    since = Column(String(20), nullable=False)
    total = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)

class DatabaseManager:
    def __init__(self):
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Handle Railway PostgreSQL URL format
        if database_url.startswith('postgresql://'):
            database_url = database_url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def init_database(self):
        """Initialize database tables and seed data"""
        logger.info("Initializing database...")
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Seed hourly stats
        self._seed_hourly_stats()
        
        # Initialize live stats if not exists
        self._init_live_stats()
        
        logger.info("Database initialized successfully")
    
    def _seed_hourly_stats(self):
        """Pre-seed hourly stats for all day/hour combinations"""
        with self.get_session() as session:
            try:
                # Check if already seeded
                existing = session.query(HourlyStats).first()
                if existing:
                    return
                
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Create entries for all day/hour combinations
                for day in days:
                    for hour in range(24):
                        stats = HourlyStats(
                            day_name=day,
                            hour=hour,
                            low_count=0,
                            med_count=0,
                            high_count=0
                        )
                        session.add(stats)
                
                session.commit()
                logger.info("Hourly stats seeded successfully")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error seeding hourly stats: {e}")
                raise
    
    def _init_live_stats(self):
        """Initialize live stats if not exists"""
        with self.get_session() as session:
            try:
                stats = session.query(LiveStats).filter_by(id=1).first()
                if not stats:
                    now = datetime.now(ZoneInfo("Africa/Algiers"))
                    stats = LiveStats(
                        id=1,
                        since=now.strftime("%Y-%m-%d %H:%M"),
                        total=0,
                        wins=0,
                        losses=0
                    )
                    session.add(stats)
                    session.commit()
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error initializing live stats: {e}")
                raise
    
    def add_crash_value(self, value: float, category: str, time: str, hour: int, day_name: str):
        """Add new crash value to history"""
        with self.get_session() as session:
            try:
                crash = CrashHistory(
                    crash_value=value,
                    category=category,
                    time=time,
                    hour=hour,
                    day_name=day_name
                )
                session.add(crash)
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding crash value: {e}")
                raise
    
    def update_hourly_stats(self, day_name: str, hour: int, category: str):
        """Update hourly statistics"""
        with self.get_session() as session:
            try:
                stats = session.query(HourlyStats).filter_by(
                    day_name=day_name, 
                    hour=hour
                ).first()
                
                if stats:
                    if category == 'low':
                        stats.low_count += 1
                    elif category == 'medium':
                        stats.med_count += 1
                    elif category == 'high':
                        stats.high_count += 1
                    
                    session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating hourly stats: {e}")
                raise
    
    def get_bot_state(self, chat_id: int) -> Optional[BotState]:
        """Get bot state for chat"""
        with self.get_session() as session:
            return session.query(BotState).filter_by(chat_id=chat_id).first()
    
    def set_bot_mode(self, chat_id: int, mode: str):
        """Set bot mode for chat"""
        with self.get_session() as session:
            try:
                state = session.query(BotState).filter_by(chat_id=chat_id).first()
                
                if state:
                    state.mode = mode
                else:
                    state = BotState(chat_id=chat_id, mode=mode)
                    session.add(state)
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error setting bot mode: {e}")
                raise
    
    def set_last_prediction(self, chat_id: int, prediction: float, time: str):
        """Set last prediction for chat"""
        with self.get_session() as session:
            try:
                state = session.query(BotState).filter_by(chat_id=chat_id).first()
                
                if state:
                    state.last_prediction = prediction
                    state.last_prediction_time = time
                else:
                    state = BotState(
                        chat_id=chat_id,
                        last_prediction=prediction,
                        last_prediction_time=time
                    )
                    session.add(state)
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error setting last prediction: {e}")
                raise
    
    def get_live_stats(self) -> LiveStats:
        """Get live performance statistics"""
        with self.get_session() as session:
            stats = session.query(LiveStats).filter_by(id=1).first()
            if not stats:
                # Create if not exists
                self._init_live_stats()
                stats = session.query(LiveStats).filter_by(id=1).first()
            return stats
    
    def update_live_stats(self, is_win: bool):
        """Update live statistics"""
        with self.get_session() as session:
            try:
                stats = session.query(LiveStats).filter_by(id=1).first()
                if stats:
                    stats.total += 1
                    if is_win:
                        stats.wins += 1
                    else:
                        stats.losses += 1
                    session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating live stats: {e}")
                raise
    
    def reset_live_stats(self):
        """Reset live statistics"""
        with self.get_session() as session:
            try:
                stats = session.query(LiveStats).filter_by(id=1).first()
                if stats:
                    now = datetime.now(ZoneInfo("Africa/Algiers"))
                    stats.since = now.strftime("%Y-%m-%d %H:%M")
                    stats.total = 0
                    stats.wins = 0
                    stats.losses = 0
                    session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error resetting live stats: {e}")
                raise
    
    def get_crash_history(self, limit: Optional[int] = None) -> List[CrashHistory]:
        """Get crash history data"""
        with self.get_session() as session:
            query = session.query(CrashHistory).order_by(CrashHistory.id.desc())
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def get_times_analysis(self) -> List[Tuple[str, int, float, int]]:
        """Get best day/hour analysis"""
        with self.get_session() as session:
            try:
                # Query hourly stats with high percentage calculation
                query = session.query(
                    HourlyStats.day_name,
                    HourlyStats.hour,
                    ((HourlyStats.high_count * 100.0) / 
                     func.nullif(HourlyStats.low_count + HourlyStats.med_count + HourlyStats.high_count, 0)
                    ).label('high_percentage'),
                    (HourlyStats.low_count + HourlyStats.med_count + HourlyStats.high_count).label('total')
                ).filter(
                    (HourlyStats.low_count + HourlyStats.med_count + HourlyStats.high_count) > 0
                ).order_by(
                    text('high_percentage DESC')
                )
                
                results = []
                for row in query.all():
                    results.append((
                        row.day_name,
                        row.hour,
                        row.high_percentage or 0.0,
                        row.total
                    ))
                
                return results
                
            except Exception as e:
                logger.error(f"Error getting times analysis: {e}")
                return []
    
    def get_recent_crash_values(self, limit: int = 50) -> List[float]:
        """Get recent crash values for ML features"""
        with self.get_session() as session:
            values = session.query(CrashHistory.crash_value)\
                          .order_by(CrashHistory.id.desc())\
                          .limit(limit).all()
            return [v[0] for v in values]
    
    def get_category_counts(self, day_name: str = None, hour: int = None) -> dict:
        """Get category counts for specific day/hour or overall"""
        with self.get_session() as session:
            if day_name and hour is not None:
                stats = session.query(HourlyStats).filter_by(
                    day_name=day_name, 
                    hour=hour
                ).first()
                
                if stats:
                    return {
                        'low': stats.low_count,
                        'medium': stats.med_count,
                        'high': stats.high_count
                    }
            
            # Overall counts
            result = session.query(
                func.sum(HourlyStats.low_count).label('low'),
                func.sum(HourlyStats.med_count).label('med'),
                func.sum(HourlyStats.high_count).label('high')
            ).first()
            
            return {
                'low': result.low or 0,
                'medium': result.med or 0,
                'high': result.high or 0
            }
