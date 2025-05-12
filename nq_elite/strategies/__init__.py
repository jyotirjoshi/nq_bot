#!/usr/bin/env python3
"""
Strategies Package for NQ Alpha Elite

This package provides trading strategies for NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging

# Import strategy factory
from nq_alpha_elite.strategies.strategy_factory import strategy_factory

# Import strategies
from nq_alpha_elite.strategies.trend_following import (
    MovingAverageCrossover,
    MACDStrategy,
    ADXTrendStrategy
)

from nq_alpha_elite.strategies.mean_reversion import (
    RSIStrategy,
    BollingerBandsStrategy,
    StochasticStrategy
)

from nq_alpha_elite.strategies.breakout import (
    DonchianChannelBreakout,
    VolatilityBreakout,
    PriceChannelBreakout
)

from nq_alpha_elite.strategies.volatility import (
    ATRChannelStrategy,
    VolatilityExpansionStrategy,
    KeltnerChannelStrategy
)

from nq_alpha_elite.strategies.pattern import (
    EngulfingPatternStrategy,
    DojiPatternStrategy,
    HammerPatternStrategy
)

from nq_alpha_elite.strategies.multi_timeframe import (
    MTFTrendStrategy,
    MTFMomentumStrategy
)

from nq_alpha_elite.strategies.machine_learning import (
    RandomForestStrategy,
    GradientBoostingStrategy
)

from nq_alpha_elite.strategies.hybrid_strategy import (
    TechnicalRLHybridStrategy,
    RegimeSwitchingStrategy
)

from nq_alpha_elite.strategies.specialized import (
    GapFillStrategy,
    VolumeBreakoutStrategy,
    SwingHighLowStrategy,
    MarketOpenStrategy
)

from nq_alpha_elite.strategies.advanced import (
    IchimokuCloudStrategy,
    ElliottWaveStrategy,
    FibonacciRetracementStrategy
)

from nq_alpha_elite.strategies.statistical import (
    MeanReversionZScoreStrategy,
    StatisticalArbitrageStrategy,
    KalmanFilterStrategy,
    SeasonalityStrategy
)

from nq_alpha_elite.strategies.sentiment import (
    MarketSentimentStrategy,
    NewsEventStrategy,
    SocialMediaSentimentStrategy,
    OptionFlowStrategy
)

# Register trend following strategies
strategy_factory.register_strategy(MovingAverageCrossover, 'trend_following')
strategy_factory.register_strategy(MACDStrategy, 'trend_following')
strategy_factory.register_strategy(ADXTrendStrategy, 'trend_following')

# Register mean reversion strategies
strategy_factory.register_strategy(RSIStrategy, 'mean_reversion')
strategy_factory.register_strategy(BollingerBandsStrategy, 'mean_reversion')
strategy_factory.register_strategy(StochasticStrategy, 'mean_reversion')

# Register breakout strategies
strategy_factory.register_strategy(DonchianChannelBreakout, 'breakout')
strategy_factory.register_strategy(VolatilityBreakout, 'breakout')
strategy_factory.register_strategy(PriceChannelBreakout, 'breakout')

# Register volatility strategies
strategy_factory.register_strategy(ATRChannelStrategy, 'volatility')
strategy_factory.register_strategy(VolatilityExpansionStrategy, 'volatility')
strategy_factory.register_strategy(KeltnerChannelStrategy, 'volatility')

# Register pattern strategies
strategy_factory.register_strategy(EngulfingPatternStrategy, 'pattern')
strategy_factory.register_strategy(DojiPatternStrategy, 'pattern')
strategy_factory.register_strategy(HammerPatternStrategy, 'pattern')

# Register multi-timeframe strategies
strategy_factory.register_strategy(MTFTrendStrategy, 'multi_timeframe')
strategy_factory.register_strategy(MTFMomentumStrategy, 'multi_timeframe')

# Register machine learning strategies
strategy_factory.register_strategy(RandomForestStrategy, 'machine_learning')
strategy_factory.register_strategy(GradientBoostingStrategy, 'machine_learning')

# Register hybrid strategies
strategy_factory.register_strategy(TechnicalRLHybridStrategy, 'hybrid')
strategy_factory.register_strategy(RegimeSwitchingStrategy, 'hybrid')

# Register specialized strategies
strategy_factory.register_strategy(GapFillStrategy, 'specialized')
strategy_factory.register_strategy(VolumeBreakoutStrategy, 'specialized')
strategy_factory.register_strategy(SwingHighLowStrategy, 'specialized')
strategy_factory.register_strategy(MarketOpenStrategy, 'specialized')

# Register advanced strategies
strategy_factory.register_strategy(IchimokuCloudStrategy, 'advanced')
strategy_factory.register_strategy(ElliottWaveStrategy, 'advanced')
strategy_factory.register_strategy(FibonacciRetracementStrategy, 'advanced')

# Register statistical strategies
strategy_factory.register_strategy(MeanReversionZScoreStrategy, 'statistical')
strategy_factory.register_strategy(StatisticalArbitrageStrategy, 'statistical')
strategy_factory.register_strategy(KalmanFilterStrategy, 'statistical')
strategy_factory.register_strategy(SeasonalityStrategy, 'statistical')

# Register sentiment strategies
strategy_factory.register_strategy(MarketSentimentStrategy, 'sentiment')
strategy_factory.register_strategy(NewsEventStrategy, 'sentiment')
strategy_factory.register_strategy(SocialMediaSentimentStrategy, 'sentiment')
strategy_factory.register_strategy(OptionFlowStrategy, 'sentiment')

# Auto-discover any additional strategies
strategy_factory.auto_discover_strategies()

# Log registered strategies
logger = logging.getLogger("NQAlpha.Strategies")
logger.info(f"Registered {len(strategy_factory.strategies)} strategies in {len(strategy_factory.categories)} categories")