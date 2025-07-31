import pandas as pd
import numpy as np
import requests
import time
import json
import os
import hmac
import hashlib
import base64
import urllib.parse
import traceback
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import signal
import sys
from dotenv import load_dotenv
import uuid

# Web server imports
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("multi_asset_bot.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class BTCConfig:
    """BTC trading configuration"""
    # POSITION SIZING
    initial_position_usd: float = 25.0       # $25 initial entries
    stack_entry_usd: float = 20.0            # $20 additional entries
    max_stack_entries: int = 6               # Up to 6 entries per stack
    max_stack_usd: float = 125.0             # Max $125 per stack
    max_concurrent_stacks: int = 2           # 2 concurrent BTC positions
    
    # PROFIT TAKING
    profit_level_1_pct: float = 0.015        # 1.5% - Take 25% of position
    profit_level_2_pct: float = 0.025        # 2.5% - Take 50% of position  
    profit_level_3_pct: float = 0.040        # 4.0% - Take remaining 25%
    
    # STACKING STRATEGY
    stack_spacing_pct: float = 0.012         # Stack every 1.2% drop
    
    # RISK MANAGEMENT
    stop_loss_enabled: bool = False          # No stop loss for BTC (optional)
    stop_loss_pct: float = 0.08              # 8% stop loss if enabled

@dataclass
class SOLConfig:
    """SOL trading configuration - tighter due to higher volatility"""
    # POSITION SIZING (smaller due to volatility)
    initial_position_usd: float = 15.0       # $15 initial entries
    stack_entry_usd: float = 12.0            # $12 additional entries
    max_stack_entries: int = 5               # Up to 5 entries per stack
    max_stack_usd: float = 63.0              # Max $63 per stack
    max_concurrent_stacks: int = 2           # 2 concurrent SOL positions
    
    # PROFIT TAKING (tighter levels)
    profit_level_1_pct: float = 0.010        # 1.0% - Take 25% of position
    profit_level_2_pct: float = 0.020        # 2.0% - Take 50% of position  
    profit_level_3_pct: float = 0.035        # 3.5% - Take remaining 25%
    
    # STACKING STRATEGY (more aggressive)
    stack_spacing_pct: float = 0.010         # Stack every 1.0% drop
    
    # RISK MANAGEMENT (mandatory stop loss)
    stop_loss_enabled: bool = True           # Always use stop loss for SOL
    stop_loss_pct: float = 0.03              # 3% stop loss (tight)

@dataclass
class MultiAssetConfig:
    """Multi-asset bot configuration"""
    # CAPITAL ALLOCATION
    total_available_usd: float = 450.0
    btc_allocation_pct: float = 0.70         # 70% to BTC
    sol_allocation_pct: float = 0.30         # 30% to SOL
    
    # Asset configs
    btc: BTCConfig = field(default_factory=BTCConfig)
    sol: SOLConfig = field(default_factory=SOLConfig)
    
    # SHARED SETTINGS
    daily_loss_limit_usd: float = 25.0       # $25 max loss per day (5% of capital)
    max_consecutive_losses: int = 4
    max_trades_per_hour: int = 30
    max_daily_trades: int = 150
    cooldown_after_loss_minutes: int = 1
    update_frequency_seconds: int = 2
    
    # SIGNAL REQUIREMENTS
    min_signal_confidence: float = 0.6       # 60% confidence
    min_confirmations: int = 3               # Need 3+ confirming signals
    
    # TECHNICAL INDICATORS
    ema_fast: int = 3
    ema_slow: int = 8
    rsi_period: int = 9
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bb_period: int = 12
    bb_std: float = 2.0
    stoch_period: int = 8
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 4
    
    # ADVANCED ANALYSIS PERIODS
    support_resistance_periods: int = 100
    fibonacci_lookback: int = 50
    squeeze_threshold: float = 0.8
    fibonacci_retracement_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])
    
    # API SETTINGS
    kraken_base_url: str = "https://api.kraken.com"
    btc_pair: str = "XBTUSD"
    sol_pair: str = "SOLUSD"
    request_timeout: int = 15
    
    def get_asset_config(self, asset: str):
        """Get config for specific asset"""
        return self.btc if asset == "BTC" else self.sol
    
    def get_allocated_capital(self, asset: str) -> float:
        """Get allocated capital for asset"""
        if asset == "BTC":
            return self.total_available_usd * self.btc_allocation_pct
        else:
            return self.total_available_usd * self.sol_allocation_pct

@dataclass
class MultiAssetPosition:
    """Position that tracks asset type"""
    position_id: str
    asset: str  # "BTC" or "SOL"
    entries: List[Dict]
    total_usd_invested: float
    total_amount: float  # BTC or SOL amount
    average_cost_basis: float
    first_entry_time: datetime
    last_entry_time: datetime
    
    # Advanced tracking
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)
    support_resistance: Dict[str, float] = field(default_factory=dict)
    profit_levels_hit: List[str] = field(default_factory=list)
    
    # Profit taking tracking
    original_amount: float = 0.0
    amount_sold_at_levels: float = 0.0
    
    # Stop loss tracking
    stop_loss_price: float = 0.0
    stop_loss_enabled: bool = False
    
    # Fee tracking
    kraken_taker_fee: float = 0.0026
    
    def __post_init__(self):
        if self.original_amount == 0.0:
            self.original_amount = self.total_amount
    
    def add_entry(self, price: float, usd_amount: float, asset_amount: float):
        """Add another entry to the stack"""
        entry = {
            'time': datetime.now(),
            'price': price,
            'usd_amount': usd_amount,
            'asset_amount': asset_amount
        }
        self.entries.append(entry)
        
        # Update totals
        self.total_usd_invested += usd_amount
        self.total_amount += asset_amount
        self.average_cost_basis = self.total_usd_invested / self.total_amount if self.total_amount > 0 else price
        self.last_entry_time = datetime.now()
    
    def get_profit_level_price(self, level: int, config: MultiAssetConfig) -> float:
        """Get price for profit level"""
        asset_config = config.get_asset_config(self.asset)
        if level == 1:
            return self.average_cost_basis * (1 + asset_config.profit_level_1_pct)
        elif level == 2:
            return self.average_cost_basis * (1 + asset_config.profit_level_2_pct)
        elif level == 3:
            return self.average_cost_basis * (1 + asset_config.profit_level_3_pct)
        return self.average_cost_basis
    
    def update_stop_loss(self, config: MultiAssetConfig):
        """Update stop loss price"""
        asset_config = config.get_asset_config(self.asset)
        if asset_config.stop_loss_enabled:
            self.stop_loss_enabled = True
            self.stop_loss_price = self.average_cost_basis * (1 - asset_config.stop_loss_pct)
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        if not self.stop_loss_enabled:
            return False
        return current_price <= self.stop_loss_price
    
    def get_amount_for_level(self, level: int) -> float:
        """Get amount to sell at each profit level"""
        if level == 1:
            return self.original_amount * 0.25  # 25%
        elif level == 2:
            return self.original_amount * 0.50  # 50%  
        elif level == 3:
            return self.total_amount  # Remaining amount
        return 0.0
    
    def can_take_profit_at_level(self, level: int, current_price: float, config: MultiAssetConfig) -> bool:
        """Check if we can take profit at specific level"""
        level_str = f"level_{level}"
        
        # Already hit this level
        if level_str in self.profit_levels_hit:
            return False
        
        # Price must be above profit level
        target_price = self.get_profit_level_price(level, config)
        if current_price < target_price:
            return False
        
        # Must have enough remaining
        amount_needed = self.get_amount_for_level(level)
        if self.total_amount < amount_needed:
            return False
        
        return True
    
    def execute_profit_level(self, level: int, actual_sale_price: float):
        """Record profit level execution"""
        level_str = f"level_{level}"
        self.profit_levels_hit.append(level_str)
        
        amount_sold = self.get_amount_for_level(level)
        self.total_amount -= amount_sold
        self.amount_sold_at_levels += amount_sold
        
        logging.info(f"Profit level {level} hit: sold {amount_sold:.6f} {self.asset} at ${actual_sale_price:.2f}")

class MultiAssetKrakenAPI:
    """Kraken API with multi-asset support"""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.api_secret = os.getenv('KRAKEN_API_SECRET')
        
        self.base_url = config.kraken_base_url
        
        # Create persistent session
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Multi-Asset-Bot/1.0'})
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5
        
        print(f"✅ Multi-Asset Kraken API initialized")
    
    def _rate_limit(self):
        """Rate limiting"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def get_current_price(self, asset: str = "BTC") -> Optional[float]:
        """Get current price for asset"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/0/public/Ticker"
            pair = self.config.btc_pair if asset == "BTC" else self.config.sol_pair
            params = {"pair": pair}
            
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            
            if response.status_code != 200:
                logging.error(f"HTTP {response.status_code} getting {asset} price")
                return None
            
            data = response.json()
            
            if data.get('error'):
                logging.error(f"API error: {data['error']}")
                return None
            
            result = data.get('result', {})
            for pair_key, pair_data in result.items():
                if 'c' in pair_data and pair_data['c'][0]:
                    price = float(pair_data['c'][0])
                    # Validate price ranges
                    if asset == "BTC" and 10000 <= price <= 500000:
                        return price
                    elif asset == "SOL" and 1 <= price <= 1000:
                        return price
            
            return None
            
        except Exception as e:
            logging.error(f"{asset} price fetch error: {e}")
            return None
    
    def get_account_balances(self) -> Dict[str, float]:
        """Get all relevant balances"""
        if not self.api_key or not self.api_secret:
            return {"USD": 0, "BTC": 0, "SOL": 0}
        
        self._rate_limit()
        
        try:
            uri_path = '/0/private/Balance'
            nonce = str(int(time.time() * 1000000))
            
            data = {'nonce': nonce}
            postdata = urllib.parse.urlencode(data)
            encoded = (nonce + postdata).encode()
            message = uri_path.encode() + hashlib.sha256(encoded).digest()
            
            mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
            signature = base64.b64encode(mac.digest()).decode()
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature
            }
            
            response = self.session.post(
                self.base_url + uri_path,
                headers=headers,
                data=data,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('error'):
                    logging.error(f"Balance error: {result['error']}")
                    return {"USD": 0, "BTC": 0, "SOL": 0}
                
                balances = result.get('result', {})
                
                # Extract balances
                usd_balance = 0.0
                for key in ['ZUSD', 'USD', 'KUSD']:
                    if key in balances:
                        usd_balance = float(balances[key])
                        break
                
                btc_balance = 0.0
                for key in ['XXBT', 'XBT', 'BTC', 'ZBTC']:
                    if key in balances:
                        btc_balance = float(balances[key])
                        break
                
                sol_balance = 0.0
                for key in ['SOL', 'ZSOL', 'XSOL']:
                    if key in balances:
                        sol_balance = float(balances[key])
                        break
                
                return {
                    "USD": usd_balance,
                    "BTC": btc_balance,
                    "SOL": sol_balance
                }
            
            return {"USD": 0, "BTC": 0, "SOL": 0}
            
        except Exception as e:
            logging.error(f"Balance fetch error: {e}")
            return {"USD": 0, "BTC": 0, "SOL": 0}
    
    def place_market_buy_usd(self, usd_amount: float, asset: str = "BTC") -> Optional[str]:
        """Place market buy order for specific asset"""
        if not self.api_key or not self.api_secret:
            logging.error("No API credentials")
            return None
        
        # Safety checks
        min_order = 10 if asset == "BTC" else 5  # Lower minimum for SOL
        if usd_amount < min_order:
            logging.error(f"Position too small: ${usd_amount:.2f}")
            return None
        
        self._rate_limit()
        
        try:
            uri_path = '/0/private/AddOrder'
            nonce = str(int(time.time() * 1000000))
            
            pair = self.config.btc_pair if asset == "BTC" else self.config.sol_pair
            
            order_data = {
                'nonce': nonce,
                'pair': pair,
                'type': 'buy',
                'ordertype': 'market',
                'volume': f"{usd_amount:.2f}",
                'oflags': 'viqc'  # volume in quote currency (USD)
            }
            
            postdata = urllib.parse.urlencode(order_data)
            encoded = (nonce + postdata).encode()
            message = uri_path.encode() + hashlib.sha256(encoded).digest()
            
            mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
            signature = base64.b64encode(mac.digest()).decode()
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature
            }
            
            response = self.session.post(
                self.base_url + uri_path,
                headers=headers,
                data=order_data,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('error'):
                    logging.error(f"Buy order error: {result['error']}")
                    return None
                
                order_ids = result.get('result', {}).get('txid', [])
                if order_ids:
                    logging.info(f"✅ {asset} Buy order: ${usd_amount:.2f} (ID: {order_ids[0]})")
                    return order_ids[0]
            
            return None
            
        except Exception as e:
            logging.error(f"Buy order error: {e}")
            return None
    
    def place_market_sell(self, amount: float, asset: str = "BTC") -> Optional[str]:
        """Place market sell order"""
        if not self.api_key or not self.api_secret:
            logging.error("No API credentials")
            return None
        
        min_amount = 0.00001 if asset == "BTC" else 0.01
        if amount < min_amount:
            logging.error(f"{asset} amount too small: {amount:.8f}")
            return None
        
        self._rate_limit()
        
        try:
            uri_path = '/0/private/AddOrder'
            nonce = str(int(time.time() * 1000000))
            
            pair = self.config.btc_pair if asset == "BTC" else self.config.sol_pair
            
            order_data = {
                'nonce': nonce,
                'pair': pair,
                'type': 'sell',
                'ordertype': 'market',
                'volume': f"{amount:.8f}"
            }
            
            postdata = urllib.parse.urlencode(order_data)
            encoded = (nonce + postdata).encode()
            message = uri_path.encode() + hashlib.sha256(encoded).digest()
            
            mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
            signature = base64.b64encode(mac.digest()).decode()
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature
            }
            
            response = self.session.post(
                self.base_url + uri_path,
                headers=headers,
                data=order_data,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('error'):
                    logging.error(f"Sell order error: {result['error']}")
                    return None
                
                order_ids = result.get('result', {}).get('txid', [])
                if order_ids:
                    logging.info(f"✅ {asset} Sell order: {amount:.8f} (ID: {order_ids[0]})")
                    return order_ids[0]
            
            return None
            
        except Exception as e:
            logging.error(f"Sell order error: {e}")
            return None

# Import the technical analysis classes from the original script
class AdvancedTechnicalAnalysis:
    """Advanced technical analysis with Fibonacci, S/R, and Bollinger squeeze"""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
    
    def calculate_fibonacci_levels(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        if len(prices) < self.config.fibonacci_lookback:
            return {}
        
        # Get recent high and low
        recent_prices = prices[-self.config.fibonacci_lookback:]
        swing_high = np.max(recent_prices)
        swing_low = np.min(recent_prices)
        
        if swing_high == swing_low:
            return {}
        
        # Calculate Fibonacci levels
        range_diff = swing_high - swing_low
        fib_levels = {}
        
        for level in self.config.fibonacci_retracement_levels:
            fib_price = swing_high - (range_diff * level)
            fib_levels[f"fib_{level:.3f}"] = fib_price
        
        fib_levels['swing_high'] = swing_high
        fib_levels['swing_low'] = swing_low
        
        return fib_levels
    
    def find_support_resistance(self, prices: np.ndarray, current_price: float) -> Dict[str, float]:
        """Find key support and resistance levels"""
        if len(prices) < self.config.support_resistance_periods:
            return {}
        
        recent_prices = prices[-self.config.support_resistance_periods:]
        
        # Find local peaks (resistance) and valleys (support)
        peaks = []
        valleys = []
        
        for i in range(2, len(recent_prices) - 2):
            # Peak detection
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i-2] and 
                recent_prices[i] > recent_prices[i+2]):
                peaks.append(recent_prices[i])
            
            # Valley detection
            if (recent_prices[i] < recent_prices[i-1] and 
                recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i-2] and 
                recent_prices[i] < recent_prices[i+2]):
                valleys.append(recent_prices[i])
        
        levels = {}
        
        # Find nearest support (below current price)
        valid_supports = [v for v in valleys if v < current_price * 0.98]  # At least 2% below
        if valid_supports:
            levels['nearest_support'] = max(valid_supports)
            levels['strong_support'] = np.median(valid_supports) if len(valid_supports) > 2 else max(valid_supports)
        
        # Find nearest resistance (above current price)
        valid_resistances = [p for p in peaks if p > current_price * 1.02]  # At least 2% above
        if valid_resistances:
            levels['nearest_resistance'] = min(valid_resistances)
            levels['strong_resistance'] = np.median(valid_resistances) if len(valid_resistances) > 2 else min(valid_resistances)
        
        return levels
    
    def detect_bollinger_squeeze(self, prices: np.ndarray) -> Dict[str, float]:
        """Detect Bollinger Band squeeze for breakout potential"""
        if len(prices) < self.config.bb_period + 10:
            return {}
        
        # Calculate Bollinger Bands
        bb = BollingerBands(pd.Series(prices), window=self.config.bb_period, window_dev=self.config.bb_std)
        
        upper_band = bb.bollinger_hband().values
        lower_band = bb.bollinger_lband().values
        middle_band = bb.bollinger_mavg().values
        
        if len(upper_band) < 20:
            return {}
        
        # Calculate band width (normalized)
        current_width = (upper_band[-1] - lower_band[-1]) / middle_band[-1]
        avg_width = np.mean([(upper_band[i] - lower_band[i]) / middle_band[i] 
                            for i in range(-20, -1)])
        
        squeeze_ratio = current_width / avg_width if avg_width > 0 else 1
        
        # Detect squeeze (bands contracting)
        is_squeeze = squeeze_ratio < self.config.squeeze_threshold
        
        # Calculate position within bands
        current_price = prices[-1]
        band_position = (current_price - lower_band[-1]) / (upper_band[-1] - lower_band[-1])
        
        return {
            'is_squeeze': is_squeeze,
            'squeeze_ratio': squeeze_ratio,
            'band_position': band_position,
            'upper_band': upper_band[-1],
            'lower_band': lower_band[-1],
            'middle_band': middle_band[-1],
            'width_percentile': squeeze_ratio
        }

class EnhancedSignalGenerator:
    """Enhanced signal generator with advanced technical analysis"""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.technical_analysis = AdvancedTechnicalAnalysis(config)
    
    def generate_enhanced_signals(self, price_history: List[float], asset: str = "BTC") -> Dict:
        """Generate enhanced trading signals"""
        if len(price_history) < 60:
            return {'action': 'hold', 'confidence': 0, 'reason': 'Insufficient data'}
        
        try:
            prices = np.array(price_history)
            current_price = prices[-1]
            signals = []
            confirmations = []
            
            # 1. Fibonacci Retracement Analysis
            fib_signals = self._fibonacci_analysis(prices, current_price)
            signals.extend(fib_signals['signals'])
            confirmations.extend(fib_signals['confirmations'])
            
            # 2. Support/Resistance Analysis
            sr_signals = self._support_resistance_analysis(prices, current_price)
            signals.extend(sr_signals['signals'])
            confirmations.extend(sr_signals['confirmations'])
            
            # 3. Bollinger Squeeze Analysis
            bb_signals = self._bollinger_squeeze_analysis(prices, current_price)
            signals.extend(bb_signals['signals'])
            confirmations.extend(bb_signals['confirmations'])
            
            # 4. Enhanced Momentum Analysis
            momentum_signals = self._enhanced_momentum_analysis(prices)
            signals.extend(momentum_signals['signals'])
            confirmations.extend(momentum_signals['confirmations'])
            
            # 5. Multi-timeframe RSI
            rsi_signals = self._multi_rsi_analysis(prices)
            signals.extend(rsi_signals['signals'])
            confirmations.extend(rsi_signals['confirmations'])
            
            # 6. MACD with Volume Confirmation
            macd_signals = self._macd_volume_analysis(prices)
            signals.extend(macd_signals['signals'])
            confirmations.extend(macd_signals['confirmations'])
            
            # Combine all signals
            combined_signal = self._combine_enhanced_signals(signals, confirmations)
            
            return combined_signal
            
        except Exception as e:
            logging.error(f"Enhanced signal error: {e}")
            return {'action': 'hold', 'confidence': 0, 'reason': f'Error: {e}'}
    
    def _fibonacci_analysis(self, prices: np.ndarray, current_price: float) -> Dict:
        """Fibonacci retracement analysis"""
        signals = []
        confirmations = []
        
        fib_levels = self.technical_analysis.calculate_fibonacci_levels(prices)
        
        if not fib_levels:
            return {'signals': signals, 'confirmations': confirmations}
        
        # Check if price is near key Fibonacci levels
        for level_name, level_price in fib_levels.items():
            if level_name.startswith('fib_'):
                level_pct = float(level_name.split('_')[1])
                price_diff_pct = abs(current_price - level_price) / current_price
                
                if price_diff_pct < 0.005:  # Within 0.5% of Fib level
                    # Golden ratio levels (0.618, 0.5) are stronger
                    if level_pct in [0.618, 0.5]:
                        if current_price > level_price:
                            signals.append('long')
                            confirmations.append(f"Fib {level_pct:.1%} support")
                        else:
                            signals.append('short')
                            confirmations.append(f"Fib {level_pct:.1%} resistance")
                    # Other levels are weaker signals
                    elif level_pct in [0.382, 0.236]:
                        if current_price > level_price:
                            signals.append('long')
                            confirmations.append(f"Fib {level_pct:.1%} bounce")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _support_resistance_analysis(self, prices: np.ndarray, current_price: float) -> Dict:
        """Support and resistance analysis"""
        signals = []
        confirmations = []
        
        sr_levels = self.technical_analysis.find_support_resistance(prices, current_price)
        
        if not sr_levels:
            return {'signals': signals, 'confirmations': confirmations}
        
        # Check distance to support/resistance
        if 'nearest_support' in sr_levels:
            support = sr_levels['nearest_support']
            distance_to_support = (current_price - support) / current_price
            
            if distance_to_support < 0.015:  # Within 1.5% of support
                signals.append('long')
                confirmations.append(f"Near support ${support:,.0f}")
        
        if 'nearest_resistance' in sr_levels:
            resistance = sr_levels['nearest_resistance']
            distance_to_resistance = (resistance - current_price) / current_price
            
            if distance_to_resistance < 0.015:  # Within 1.5% of resistance
                signals.append('short')
                confirmations.append(f"Near resistance ${resistance:,.0f}")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _bollinger_squeeze_analysis(self, prices: np.ndarray, current_price: float) -> Dict:
        """Bollinger Band squeeze analysis"""
        signals = []
        confirmations = []
        
        bb_data = self.technical_analysis.detect_bollinger_squeeze(prices)
        
        if not bb_data:
            return {'signals': signals, 'confirmations': confirmations}
        
        is_squeeze = bb_data.get('is_squeeze', False)
        band_position = bb_data.get('band_position', 0.5)
        squeeze_ratio = bb_data.get('squeeze_ratio', 1.0)
        
        if is_squeeze:
            # During squeeze, look for breakout direction
            if band_position > 0.7:  # Near upper band
                signals.append('long')
                confirmations.append(f"BB squeeze breakout up")
            elif band_position < 0.3:  # Near lower band
                signals.append('short')
                confirmations.append(f"BB squeeze breakout down")
            
            # Add squeeze strength to confirmations
            if squeeze_ratio < 0.6:
                confirmations.append(f"Strong BB compression")
        else:
            # Normal BB signals
            if band_position > 0.8:
                signals.append('short')
                confirmations.append(f"BB overbought")
            elif band_position < 0.2:
                signals.append('long')
                confirmations.append(f"BB oversold")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _enhanced_momentum_analysis(self, prices: np.ndarray) -> Dict:
        """Enhanced momentum analysis"""
        signals = []
        confirmations = []
        
        if len(prices) < 20:
            return {'signals': signals, 'confirmations': confirmations}
        
        # Multiple timeframe EMAs
        ema_3 = self._ema(prices, 3)
        ema_8 = self._ema(prices, 8)
        ema_21 = self._ema(prices, 21)
        
        current_price = prices[-1]
        
        # EMA alignment
        if ema_3 > ema_8 > ema_21:
            signals.append('long')
            confirmations.append("EMA bullish alignment")
        elif ema_3 < ema_8 < ema_21:
            signals.append('short')
            confirmations.append("EMA bearish alignment")
        
        # Price vs EMA position
        if current_price > ema_8 * 1.002:
            signals.append('long')
            confirmations.append("Above EMA8")
        elif current_price < ema_8 * 0.998:
            signals.append('short')
            confirmations.append("Below EMA8")
        
        # Momentum acceleration
        if len(prices) >= 10:
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]
            older_momentum = (prices[-5] - prices[-10]) / prices[-10]
            
            if recent_momentum > older_momentum and recent_momentum > 0.003:
                signals.append('long')
                confirmations.append("Momentum accelerating up")
            elif recent_momentum < older_momentum and recent_momentum < -0.003:
                signals.append('short')
                confirmations.append("Momentum accelerating down")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _multi_rsi_analysis(self, prices: np.ndarray) -> Dict:
        """Multi-timeframe RSI analysis"""
        signals = []
        confirmations = []
        
        if len(prices) < 30:
            return {'signals': signals, 'confirmations': confirmations}
        
        # Multiple RSI periods
        rsi_7 = self._rsi(prices, 7)
        rsi_14 = self._rsi(prices, 14)
        rsi_21 = self._rsi(prices, 21)
        
        # RSI convergence signals
        if rsi_7 < 30 and rsi_14 < 35:
            signals.append('long')
            confirmations.append(f"Multi RSI oversold")
        elif rsi_7 > 70 and rsi_14 > 65:
            signals.append('short')
            confirmations.append(f"Multi RSI overbought")
        
        # RSI divergence (simplified)
        if rsi_7 > rsi_14 > rsi_21 and rsi_14 > 50:
            signals.append('long')
            confirmations.append("RSI momentum building")
        elif rsi_7 < rsi_14 < rsi_21 and rsi_14 < 50:
            signals.append('short')
            confirmations.append("RSI momentum declining")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _macd_volume_analysis(self, prices: np.ndarray) -> Dict:
        """MACD with volume confirmation"""
        signals = []
        confirmations = []
        
        if len(prices) < 30:
            return {'signals': signals, 'confirmations': confirmations}
        
        # Calculate MACD
        ema_fast = self._ema(prices, self.config.macd_fast)
        ema_slow = self._ema(prices, self.config.macd_slow)
        macd_line = ema_fast - ema_slow
        
        # Simple MACD signals
        if len(prices) >= 35:
            prev_ema_fast = self._ema(prices[:-1], self.config.macd_fast)
            prev_ema_slow = self._ema(prices[:-1], self.config.macd_slow)
            prev_macd = prev_ema_fast - prev_ema_slow
            
            # MACD crossover
            if macd_line > 0 and prev_macd <= 0:
                signals.append('long')
                confirmations.append("MACD bullish cross")
            elif macd_line < 0 and prev_macd >= 0:
                signals.append('short')
                confirmations.append("MACD bearish cross")
            
            # MACD momentum
            if macd_line > prev_macd and macd_line > 0:
                signals.append('long')
                confirmations.append("MACD momentum up")
            elif macd_line < prev_macd and macd_line < 0:
                signals.append('short')
                confirmations.append("MACD momentum down")
        
        return {'signals': signals, 'confirmations': confirmations}
    
    def _combine_enhanced_signals(self, signals: List[str], confirmations: List[str]) -> Dict:
        """Combine enhanced signals"""
        long_signals = signals.count('long')
        short_signals = signals.count('short')
        total_signals = len(signals)
        
        if total_signals == 0:
            return {'action': 'hold', 'confidence': 0, 'reason': 'No signals generated'}
        
        # Require minimum confirmations
        if long_signals >= self.config.min_confirmations and long_signals > short_signals:
            confidence = long_signals / total_signals
            
            if confidence >= self.config.min_signal_confidence:
                return {
                    'action': 'long',
                    'confidence': confidence,
                    'confirmations': confirmations[:6],  # Top 6 confirmations
                    'signal_count': long_signals,
                    'total_signals': total_signals
                }
        
        elif short_signals >= self.config.min_confirmations and short_signals > long_signals:
            confidence = short_signals / total_signals
            
            if confidence >= self.config.min_signal_confidence:
                return {
                    'action': 'short',
                    'confidence': confidence,
                    'confirmations': confirmations[:6],
                    'signal_count': short_signals,
                    'total_signals': total_signals
                }
        
        return {
            'action': 'hold',
            'confidence': max(long_signals, short_signals) / total_signals if total_signals > 0 else 0,
            'reason': f'Weak signals (L:{long_signals}, S:{short_signals})',
            'confirmations': confirmations[:3]
        }
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2.0 / (period + 1.0)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

class MultiAssetWebServer:
    """Web server for multi-asset bot dashboard"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.app = Flask(__name__)
        CORS(self.app, origins=['http://localhost:5000', 'http://127.0.0.1:5000'])
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/')
        def index():
            """Serve the dashboard HTML"""
            try:
                with open('multi_asset_dashboard.html', 'r') as f:
                    return f.read()
            except:
                return "Dashboard HTML not found. Please save the dashboard HTML as 'multi_asset_dashboard.html'"
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current bot metrics"""
            try:
                btc_price = self.bot.btc_price_history[-1] if self.bot.btc_price_history else 0
                sol_price = self.bot.sol_price_history[-1] if self.bot.sol_price_history else 0
                
                # Calculate price changes
                btc_change_pct = 0
                if len(self.bot.btc_price_history) > 1440:
                    btc_24h_ago = self.bot.btc_price_history[-1440]
                    btc_change_pct = ((btc_price - btc_24h_ago) / btc_24h_ago * 100) if btc_24h_ago > 0 else 0
                
                sol_change_pct = 0
                if len(self.bot.sol_price_history) > 1440:
                    sol_24h_ago = self.bot.sol_price_history[-1440]
                    sol_change_pct = ((sol_price - sol_24h_ago) / sol_24h_ago * 100) if sol_24h_ago > 0 else 0
                
                # Get account balances
                balances = self.bot.kraken_api.get_account_balances()
                
                # Calculate metrics by asset
                btc_positions = [p for p in self.bot.positions if p.asset == "BTC"]
                sol_positions = [p for p in self.bot.positions if p.asset == "SOL"]
                
                metrics = {
                    'btc': {
                        'price': btc_price,
                        'price_change_24h': round(btc_change_pct, 2),
                        'accumulated': round(self.bot.btc_accumulated, 6),
                        'accumulation_rate': round((self.bot.btc_successful_trades / max(self.bot.btc_total_trades, 1)) * 100, 1),
                        'total_trades': self.bot.btc_total_trades,
                        'successful_trades': self.bot.btc_successful_trades,
                        'active_positions': len(btc_positions),
                        'exposure': round(sum(p.total_usd_invested for p in btc_positions), 2),
                        'allocated_capital': round(self.bot.config.get_allocated_capital("BTC"), 2)
                    },
                    'sol': {
                        'price': sol_price,
                        'price_change_24h': round(sol_change_pct, 2),
                        'accumulated': round(self.bot.sol_accumulated, 6),
                        'accumulation_rate': round((self.bot.sol_successful_trades / max(self.bot.sol_total_trades, 1)) * 100, 1),
                        'total_trades': self.bot.sol_total_trades,
                        'successful_trades': self.bot.sol_successful_trades,
                        'active_positions': len(sol_positions),
                        'exposure': round(sum(p.total_usd_invested for p in sol_positions), 2),
                        'allocated_capital': round(self.bot.config.get_allocated_capital("SOL"), 2)
                    },
                    'overall': {
                        'total_trades': self.bot.btc_total_trades + self.bot.sol_total_trades,
                        'daily_pnl': round(self.bot.daily_pnl, 2),
                        'consecutive_losses': self.bot.consecutive_losses,
                        'usd_balance': round(balances.get("USD", 0), 2),
                        'btc_balance': round(balances.get("BTC", 0), 6),
                        'sol_balance': round(balances.get("SOL", 0), 4),
                        'runtime': str(datetime.now() - self.bot.start_time).split('.')[0] if self.bot.start_time else "0:00:00",
                        'status': 'running' if self.bot.running else 'stopped'
                    }
                }
                
                return jsonify(metrics)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get active positions"""
            try:
                positions = []
                
                for pos in self.bot.positions:
                    current_price = self.bot.btc_price_history[-1] if pos.asset == "BTC" else self.bot.sol_price_history[-1]
                    
                    # Calculate unrealized P&L
                    current_value = pos.total_amount * current_price
                    unrealized_pnl = current_value - pos.total_usd_invested
                    unrealized_pnl_pct = (unrealized_pnl / pos.total_usd_invested * 100) if pos.total_usd_invested > 0 else 0
                    
                    # Calculate profit level progress
                    levels_hit = len(pos.profit_levels_hit)
                    progress_pct = (levels_hit / 3) * 100
                    
                    position_data = {
                        'id': pos.position_id,
                        'asset': pos.asset,
                        'entries': len(pos.entries),
                        'max_entries': self.bot.config.btc.max_stack_entries if pos.asset == "BTC" else self.bot.config.sol.max_stack_entries,
                        'total_usd_invested': round(pos.total_usd_invested, 2),
                        'total_amount': round(pos.total_amount, 6 if pos.asset == "BTC" else 4),
                        'average_cost': round(pos.average_cost_basis, 2),
                        'current_price': round(current_price, 2),
                        'unrealized_pnl': round(unrealized_pnl, 2),
                        'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                        'first_entry_time': pos.first_entry_time.isoformat(),
                        'last_entry_time': pos.last_entry_time.isoformat(),
                        'duration_minutes': int((datetime.now() - pos.first_entry_time).total_seconds() / 60),
                        'profit_levels_hit': pos.profit_levels_hit,
                        'progress_pct': progress_pct,
                        'profit_targets': {
                            'level_1': round(pos.get_profit_level_price(1, self.bot.config), 2),
                            'level_2': round(pos.get_profit_level_price(2, self.bot.config), 2),
                            'level_3': round(pos.get_profit_level_price(3, self.bot.config), 2)
                        },
                        'stop_loss_enabled': pos.stop_loss_enabled,
                        'stop_loss_price': round(pos.stop_loss_price, 2) if pos.stop_loss_enabled else 0,
                        'amount_sold': round(pos.amount_sold_at_levels, 6 if pos.asset == "BTC" else 4),
                        'original_amount': round(pos.original_amount, 6 if pos.asset == "BTC" else 4)
                    }
                    
                    # Add technical levels if available
                    if pos.fibonacci_levels:
                        position_data['fibonacci_levels'] = {k: round(v, 2) for k, v in pos.fibonacci_levels.items()}
                    if pos.support_resistance:
                        position_data['support_resistance'] = {k: round(v, 2) for k, v in pos.support_resistance.items()}
                    
                    positions.append(position_data)
                
                return jsonify(positions)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/indicators/<asset>')
        def get_indicators(asset):
            """Get technical indicators for specific asset"""
            try:
                asset = asset.upper()
                if asset not in ["BTC", "SOL"]:
                    return jsonify({'error': 'Invalid asset'}), 400
                
                price_history = self.bot.btc_price_history if asset == "BTC" else self.bot.sol_price_history
                
                if len(price_history) < 30:
                    return jsonify({'error': 'Insufficient data'}), 400
                
                prices = np.array(list(price_history))
                current_price = prices[-1]
                
                # Calculate indicators
                rsi = self.bot.signal_generator._rsi(prices, 14)
                ema_fast = self.bot.signal_generator._ema(prices, self.bot.config.ema_fast)
                ema_slow = self.bot.signal_generator._ema(prices, self.bot.config.ema_slow)
                
                # Bollinger Bands
                bb_data = self.bot.signal_generator.technical_analysis.detect_bollinger_squeeze(prices)
                
                # Get current signals
                signals = self.bot.signal_generator.generate_enhanced_signals(list(prices), asset)
                
                indicators = {
                    'asset': asset,
                    'rsi': {
                        'value': round(rsi, 1),
                        'oversold': self.bot.config.rsi_oversold,
                        'overbought': self.bot.config.rsi_overbought
                    },
                    'ema': {
                        'fast': round(ema_fast, 2),
                        'slow': round(ema_slow, 2),
                        'trend': 'bullish' if ema_fast > ema_slow else 'bearish'
                    },
                    'bollinger_bands': {
                        'upper': round(bb_data.get('upper_band', 0), 2),
                        'middle': round(bb_data.get('middle_band', 0), 2),
                        'lower': round(bb_data.get('lower_band', 0), 2),
                        'position': round(bb_data.get('band_position', 0.5) * 100, 1),
                        'is_squeeze': bb_data.get('is_squeeze', False),
                        'squeeze_ratio': round(bb_data.get('squeeze_ratio', 1), 2)
                    },
                    'current_signals': {
                        'action': signals.get('action', 'hold'),
                        'confidence': round(signals.get('confidence', 0), 2),
                        'confirmations': signals.get('confirmations', [])[:6],
                        'signal_count': signals.get('signal_count', 0),
                        'total_signals': signals.get('total_signals', 0)
                    },
                    'price_levels': {
                        'current': round(current_price, 2),
                        'high_24h': round(np.max(prices[-1440:]), 2) if len(prices) > 1440 else round(current_price, 2),
                        'low_24h': round(np.min(prices[-1440:]), 2) if len(prices) > 1440 else round(current_price, 2)
                    }
                }
                
                return jsonify(indicators)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history/<asset>')
        def get_history(asset):
            """Get price history for charts"""
            try:
                asset = asset.upper()
                if asset not in ["BTC", "SOL"]:
                    return jsonify({'error': 'Invalid asset'}), 400
                
                price_history = self.bot.btc_price_history if asset == "BTC" else self.bot.sol_price_history
                
                # Get last 100 price points for charting
                price_data = []
                if len(price_history) > 0:
                    prices = list(price_history)[-100:]
                    for i, price in enumerate(prices):
                        price_data.append({
                            'time': (datetime.now() - timedelta(seconds=(len(prices) - i) * 2)).isoformat(),
                            'price': round(price, 2)
                        })
                
                return jsonify(price_data)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self, host='127.0.0.1', port=5000):
        """Start the web server in a separate thread"""
        def run_server():
            self.app.run(host=host, port=port, debug=False, threaded=True)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logging.info(f"Web server started at http://{host}:{port}")

class MultiAssetTradingBot:
    """Multi-asset trading bot for BTC and SOL"""
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self.kraken_api = MultiAssetKrakenAPI(config)
        self.signal_generator = EnhancedSignalGenerator(config)
        
        # Separate price histories for each asset
        self.btc_price_history = deque(maxlen=500)
        self.sol_price_history = deque(maxlen=500)
        
        # Combined positions list
        self.positions: List[MultiAssetPosition] = []
        
        # Separate performance tracking
        self.start_time = None
        
        # BTC metrics
        self.btc_total_trades = 0
        self.btc_accumulated = 0.0
        self.btc_successful_trades = 0
        
        # SOL metrics
        self.sol_total_trades = 0
        self.sol_accumulated = 0.0
        self.sol_successful_trades = 0
        
        # Shared risk management
        self.daily_pnl = 0.0
        self.daily_fees = 0.0
        self.consecutive_losses = 0
        
        # Capital tracking
        self.btc_allocated = config.get_allocated_capital("BTC")
        self.sol_allocated = config.get_allocated_capital("SOL")
        
        # Web server (initialized later)
        self.web_server = None
        
        # Shutdown handler
        signal.signal(signal.SIGINT, self._safe_shutdown)
    
    def _safe_shutdown(self, signum, frame):
        """Safe shutdown"""
        print(f"\n🛑 MULTI-ASSET BOT SHUTDOWN INITIATED")
        self.running = False
        
        # Close all positions
        if self.positions:
            btc_price = self._get_price("BTC")
            sol_price = self._get_price("SOL")
            
            for position in self.positions[:]:
                current_price = btc_price if position.asset == "BTC" else sol_price
                if current_price:
                    self._force_close_position(position, current_price, "Emergency shutdown")
        
        self._print_final_summary()
        sys.exit(0)
    
    def _get_price(self, asset: str = "BTC") -> Optional[float]:
        """Get current price and add to history"""
        price = self.kraken_api.get_current_price(asset)
        if price:
            if asset == "BTC":
                self.btc_price_history.append(price)
            else:
                self.sol_price_history.append(price)
        return price
    
    def _get_current_exposure(self, asset: str) -> float:
        """Get current exposure for specific asset"""
        asset_positions = [p for p in self.positions if p.asset == asset]
        return sum(p.total_usd_invested for p in asset_positions)
    
    def run(self):
        """Run multi-asset trading bot"""
        print(f"\n🚀 MULTI-ASSET TRADING BOT (BTC + SOL)")
        print(f"{'='*60}")
        print(f"💰 Total Capital: ${self.config.total_available_usd}")
        print(f"📊 Allocation: BTC ${self.btc_allocated} (70%) | SOL ${self.sol_allocated} (30%)")
        print(f"🎯 Strategy: Multi-level profit taking with enhanced risk management")
        print(f"⚡ SOL Features: 3% stop loss, tighter profit levels")
        
        # Test connections
        print(f"\n🔍 Testing connections...")
        btc_price = self._get_price("BTC")
        sol_price = self._get_price("SOL")
        
        if not btc_price or not sol_price:
            print("❌ Cannot get prices - aborting")
            return
        
        print(f"✅ Connected! BTC: ${btc_price:,.0f} | SOL: ${sol_price:.2f}")
        
        # Check balances
        balances = self.kraken_api.get_account_balances()
        print(f"💰 Balances - USD: ${balances['USD']:.2f} | BTC: {balances['BTC']:.6f} | SOL: {balances['SOL']:.4f}")
        
        # Initialize price histories
        print(f"📊 Building price histories...")
        for i in range(100):
            self._get_price("BTC")
            self._get_price("SOL")
            time.sleep(0.5)
        
        self.start_time = datetime.now()
        self.running = True
        
        # Start web server
        print(f"\n🌐 Starting dashboard web server...")
        self.web_server = MultiAssetWebServer(self)
        self.web_server.start()
        print(f"✅ Dashboard available at http://localhost:5000")
        
        print(f"\n🎯 MULTI-ASSET TRADING STARTED")
        print(f"Monitoring BTC and SOL for optimal entries")
        print(f"{'='*60}\n")
        
        last_status = datetime.now()
        
        try:
            loop_count = 0
            while self.running:
                loop_count += 1
                if loop_count % 30 == 0:  # Log every 60 seconds (30 * 2 second cycles)
                    logging.info(f"Main loop still running... (cycle {loop_count})")
                
                cycle_start = time.time()
                
                # Update prices
                try:
                    btc_price = self._get_price("BTC")
                    sol_price = self._get_price("SOL")
                    
                    if not btc_price or not sol_price:
                        logging.warning("Failed to get prices, retrying...")
                        time.sleep(2)
                        continue
                    
                    # Manage existing positions
                    for position in self.positions[:]:
                        try:
                            current_price = btc_price if position.asset == "BTC" else sol_price
                            self._manage_position(position, current_price)
                        except Exception as e:
                            logging.error(f"Position management error for {position.position_id}: {e}")
                            traceback.print_exc()
                    
                    # Check for new entries
                    if len(self.btc_price_history) >= 100 and len(self.sol_price_history) >= 100:
                        # Check BTC entries
                        btc_positions = [p for p in self.positions if p.asset == "BTC"]
                        if len(btc_positions) < self.config.btc.max_concurrent_stacks:
                            try:
                                self._check_entry(btc_price, "BTC")
                            except Exception as e:
                                logging.error(f"BTC entry check error: {e}")
                                traceback.print_exc()
                        
                        # Check SOL entries
                        sol_positions = [p for p in self.positions if p.asset == "SOL"]
                        if len(sol_positions) < self.config.sol.max_concurrent_stacks:
                            try:
                                self._check_entry(sol_price, "SOL")
                            except Exception as e:
                                logging.error(f"SOL entry check error: {e}")
                                traceback.print_exc()
                    
                    # Status update every minute
                    if (datetime.now() - last_status).total_seconds() >= 60:
                        try:
                            self._print_status(btc_price, sol_price)
                            last_status = datetime.now()
                        except Exception as e:
                            logging.error(f"Status print error: {e}")
                            import traceback
                            traceback.print_exc()
                            last_status = datetime.now()  # Reset timer even on error
                    
                except Exception as e:
                    logging.error(f"Main loop error: {e}")
                    traceback.print_exc()
                    time.sleep(2)  # Wait before retrying
                    continue
                
                # Sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.update_frequency_seconds - cycle_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logging.info("Bot interrupted by user")
            self._safe_shutdown(None, None)
        except Exception as e:
            logging.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()
            # Optionally restart or continue based on error severity
            time.sleep(5)  # Wait 5 seconds before retrying
    
    def _check_entry(self, current_price: float, asset: str):
        """Check for entry opportunities"""
        try:
            # Generate signals
            price_list = list(self.btc_price_history if asset == "BTC" else self.sol_price_history)
            signals = self.signal_generator.generate_enhanced_signals(price_list, asset)
            
            if signals['action'] != 'long':  # Only long positions
                return
            
            # Risk checks
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                return
            
            if self.daily_pnl <= -self.config.daily_loss_limit_usd:
                return
            
            # Check available capital for this asset
            allocated_capital = self.config.get_allocated_capital(asset)
            current_exposure = self._get_current_exposure(asset)
            available = allocated_capital - current_exposure
            
            asset_config = self.config.get_asset_config(asset)
            if available < asset_config.initial_position_usd:
                return
            
            # Execute entry
            self._execute_entry(current_price, signals, asset)
                
        except Exception as e:
            logging.error(f"{asset} entry check error: {e}")
    
    def _execute_entry(self, price: float, signals: Dict, asset: str):
        """Execute entry with advanced analysis"""
        try:
            asset_config = self.config.get_asset_config(asset)
            position_size = asset_config.initial_position_usd
            
            # Place order
            order_id = self.kraken_api.place_market_buy_usd(position_size, asset)
            if not order_id:
                return
            
            # Calculate amount received
            estimated_amount = (position_size * 0.9974) / price  # Account for fees
            
            # Create position with technical analysis
            price_array = np.array(list(self.btc_price_history if asset == "BTC" else self.sol_price_history))
            fib_levels = self.signal_generator.technical_analysis.calculate_fibonacci_levels(price_array)
            sr_levels = self.signal_generator.technical_analysis.find_support_resistance(price_array, price)
            
            initial_entry = {
                'time': datetime.now(),
                'price': price,
                'usd_amount': position_size,
                'asset_amount': estimated_amount
            }
            
            # Update trade counter
            if asset == "BTC":
                self.btc_total_trades += 1
                position_id = f"btc_{self.btc_total_trades}"
            else:
                self.sol_total_trades += 1
                position_id = f"sol_{self.sol_total_trades}"
            
            position = MultiAssetPosition(
                position_id=position_id,
                asset=asset,
                entries=[initial_entry],
                total_usd_invested=position_size,
                total_amount=estimated_amount,
                average_cost_basis=price,
                first_entry_time=datetime.now(),
                last_entry_time=datetime.now(),
                fibonacci_levels=fib_levels,
                support_resistance=sr_levels
            )
            
            # Set stop loss for SOL
            position.update_stop_loss(self.config)
            
            self.positions.append(position)
            
            # Get profit targets for display
            profit_1 = position.get_profit_level_price(1, self.config)
            profit_2 = position.get_profit_level_price(2, self.config)
            profit_3 = position.get_profit_level_price(3, self.config)
            
            print(f"🟢 {asset} ENTRY: ${position_size:.2f} at ${price:,.2f}")
            print(f"   Position ID: {position.position_id}")
            print(f"   Confidence: {signals['confidence']:.2f}")
            print(f"   Signals: {', '.join(signals['confirmations'][:3])}")
            print(f"   Profit Levels: ${profit_1:,.2f} / ${profit_2:,.2f} / ${profit_3:,.2f}")
            if position.stop_loss_enabled:
                print(f"   Stop Loss: ${position.stop_loss_price:,.2f}")
            
        except Exception as e:
            logging.error(f"{asset} entry error: {e}")
    
    def _manage_position(self, position: MultiAssetPosition, current_price: float):
        """Manage position with stop loss and profit taking"""
        try:
            # Check stop loss first (for SOL)
            if position.should_stop_loss(current_price):
                self._execute_stop_loss(position, current_price)
                return
            
            # Check for stacking opportunities
            self._check_stacking(position, current_price)
            
            # Check profit levels
            for level in [1, 2, 3]:
                if position.can_take_profit_at_level(level, current_price, self.config):
                    self._execute_profit_level(position, level, current_price)
            
            # Check for complete exit
            if len(position.profit_levels_hit) >= 3 or position.total_amount < 0.00001:
                self._complete_position_exit(position, current_price)
                
        except Exception as e:
            logging.error(f"Position management error: {e}")
    
    def _execute_stop_loss(self, position: MultiAssetPosition, current_price: float):
        """Execute stop loss"""
        try:
            if position.total_amount <= 0:
                return
            
            # Place sell order
            order_id = self.kraken_api.place_market_sell(position.total_amount, position.asset)
            if order_id:
                loss_pct = ((current_price - position.average_cost_basis) / position.average_cost_basis) * 100
                
                print(f"🛑 STOP LOSS: {position.position_id}")
                print(f"   Sold {position.total_amount:.6f} {position.asset} at ${current_price:,.2f}")
                print(f"   Loss: {loss_pct:.1f}%")
                
                self.consecutive_losses += 1
                self.positions.remove(position)
                
        except Exception as e:
            logging.error(f"Stop loss error: {e}")
    
    def _check_stacking(self, position: MultiAssetPosition, current_price: float):
        """Check for stacking opportunities"""
        try:
            asset_config = self.config.get_asset_config(position.asset)
            
            # Don't stack if at maximum entries
            if len(position.entries) >= asset_config.max_stack_entries:
                return
            
            # Don't stack if position value is too high
            if position.total_usd_invested >= asset_config.max_stack_usd:
                return
            
            # Check if price dropped enough
            last_entry_price = position.entries[-1]['price']
            price_drop_pct = (last_entry_price - current_price) / last_entry_price
            
            if price_drop_pct >= asset_config.stack_spacing_pct:
                # Check available balance
                allocated_capital = self.config.get_allocated_capital(position.asset)
                current_exposure = self._get_current_exposure(position.asset)
                available = allocated_capital - current_exposure
                
                if available < asset_config.stack_entry_usd:
                    return
                
                # Execute stack
                order_id = self.kraken_api.place_market_buy_usd(asset_config.stack_entry_usd, position.asset)
                if order_id:
                    estimated_amount = (asset_config.stack_entry_usd * 0.9974) / current_price
                    position.add_entry(current_price, asset_config.stack_entry_usd, estimated_amount)
                    
                    # Update stop loss
                    position.update_stop_loss(self.config)
                    
                    print(f"📚 {position.asset} STACK: ${asset_config.stack_entry_usd:.2f} to {position.position_id}")
                    print(f"   Drop: {price_drop_pct*100:.1f}%")
                    print(f"   New avg: ${position.average_cost_basis:,.2f}")
                    print(f"   Entries: {len(position.entries)}/{asset_config.max_stack_entries}")
                    if position.stop_loss_enabled:
                        print(f"   New stop loss: ${position.stop_loss_price:,.2f}")
                        
        except Exception as e:
            logging.error(f"Stacking error: {e}")
    
    def _execute_profit_level(self, position: MultiAssetPosition, level: int, current_price: float):
        """Execute profit taking at specific level"""
        try:
            amount_to_sell = position.get_amount_for_level(level)
            
            if amount_to_sell <= 0:
                return
            
            # Place sell order
            order_id = self.kraken_api.place_market_sell(amount_to_sell, position.asset)
            if order_id:
                position.execute_profit_level(level, current_price)
                
                # Calculate profit
                profit_amount = amount_to_sell - (amount_to_sell * position.average_cost_basis / current_price)
                
                print(f"🎯 PROFIT LEVEL {level}: Sold {amount_to_sell:.6f} {position.asset} at ${current_price:,.2f}")
                print(f"   {position.asset} Profit: +{profit_amount:.6f}")
                print(f"   Remaining: {position.total_amount:.6f} {position.asset}")
                
                if profit_amount > 0:
                    if position.asset == "BTC":
                        self.btc_accumulated += profit_amount
                        self.btc_successful_trades += 1
                    else:
                        self.sol_accumulated += profit_amount
                        self.sol_successful_trades += 1
                    
                    self.consecutive_losses = 0  # Reset on profit
                
        except Exception as e:
            logging.error(f"Profit level execution error: {e}")
    
    def _complete_position_exit(self, position: MultiAssetPosition, current_price: float):
        """Complete position exit"""
        try:
            if position.total_amount > 0:
                order_id = self.kraken_api.place_market_sell(position.total_amount, position.asset)
                if order_id:
                    amount_sold = position.total_amount
                    position.total_amount = 0
                    
                    print(f"🏁 POSITION COMPLETE: {position.position_id}")
                    print(f"   Final sale: {amount_sold:.6f} {position.asset} at ${current_price:,.2f}")
            
            # Remove from active positions
            self.positions.remove(position)
            
        except Exception as e:
            logging.error(f"Complete exit error: {e}")
    
    def _force_close_position(self, position: MultiAssetPosition, current_price: float, reason: str):
        """Force close position during shutdown"""
        try:
            if position.total_amount > 0:
                order_id = self.kraken_api.place_market_sell(position.total_amount, position.asset)
                if order_id:
                    print(f"🛑 FORCED CLOSE: {position.position_id} - {reason}")
            
            self.positions.remove(position)
            
        except Exception as e:
            logging.error(f"Force close error: {e}")
    
    def _print_status(self, btc_price: float, sol_price: float):
        """Print status"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        print(f"\n📊 MULTI-ASSET STATUS | {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Prices: BTC ${btc_price:,.0f} | SOL ${sol_price:.2f}")
        
        # BTC Stats
        btc_positions = [p for p in self.positions if p.asset == "BTC"]
        btc_exposure = sum(p.total_usd_invested for p in btc_positions)
        btc_rate = (self.btc_successful_trades / max(self.btc_total_trades, 1)) * 100
        
        print(f"\n   📈 BTC Performance:")
        print(f"      Accumulated: {self.btc_accumulated:+.6f} BTC")
        print(f"      Success Rate: {btc_rate:.1f}%")
        print(f"      Active Positions: {len(btc_positions)}")
        print(f"      Exposure: ${btc_exposure:.2f} / ${self.btc_allocated:.2f}")
        
        # SOL Stats
        sol_positions = [p for p in self.positions if p.asset == "SOL"]
        sol_exposure = sum(p.total_usd_invested for p in sol_positions)
        sol_rate = (self.sol_successful_trades / max(self.sol_total_trades, 1)) * 100
        
        print(f"\n   ⚡ SOL Performance:")
        print(f"      Accumulated: {self.sol_accumulated:+.4f} SOL")
        print(f"      Success Rate: {sol_rate:.1f}%")
        print(f"      Active Positions: {len(sol_positions)}")
        print(f"      Exposure: ${sol_exposure:.2f} / ${self.sol_allocated:.2f}")
        
        print(f"\n   ⏱️  Runtime: {str(runtime).split('.')[0]}")
        
        if self.positions:
            print(f"\n📚 ACTIVE POSITIONS:")
            for i, pos in enumerate(self.positions, 1):
                current_price = btc_price if pos.asset == "BTC" else sol_price
                unrealized_pnl = (pos.total_amount * current_price) - pos.total_usd_invested
                unrealized_pnl_pct = (unrealized_pnl / pos.total_usd_invested * 100) if pos.total_usd_invested > 0 else 0
                
                print(f"   {i}. {pos.position_id} | Avg: ${pos.average_cost_basis:,.2f}")
                amount_str = f"{pos.total_amount:.6f}" if pos.asset == 'BTC' else f"{pos.total_amount:.4f}"
                print(f"      Entries: {len(pos.entries)} | Amount: {amount_str}")
                print(f"      P&L: ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.1f}%)")
                
                levels_hit = len(pos.profit_levels_hit)
                progress = "●" * levels_hit + "○" * (3 - levels_hit)
                print(f"      Progress: {progress} ({levels_hit}/3 levels)")
                
                if pos.stop_loss_enabled:
                    print(f"      Stop Loss: ${pos.stop_loss_price:,.2f}")
                print()
    
    def _print_final_summary(self):
        """Print final summary"""
        print(f"\n🏁 MULTI-ASSET TRADING SESSION COMPLETE")
        print(f"{'='*60}")
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"Runtime: {str(runtime).split('.')[0]}")
        
        print(f"\n📈 BTC Results:")
        print(f"   Total Trades: {self.btc_total_trades}")
        print(f"   Successful Trades: {self.btc_successful_trades}")
        print(f"   BTC Accumulated: {self.btc_accumulated:+.6f}")
        
        print(f"\n⚡ SOL Results:")
        print(f"   Total Trades: {self.sol_total_trades}")
        print(f"   Successful Trades: {self.sol_successful_trades}")
        print(f"   SOL Accumulated: {self.sol_accumulated:+.4f}")
        
        total_trades = self.btc_total_trades + self.sol_total_trades
        print(f"\n📊 Overall:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Daily P&L: ${self.daily_pnl:+.2f}")
        
        print(f"\n✅ Multi-asset trading complete")

def main():
    """Main function for multi-asset trading"""
    print("🚀 MULTI-ASSET TRADING BOT")
    print("=" * 60)
    print("BTC + SOL with Advanced Risk Management")
    
    if not os.getenv('KRAKEN_API_KEY') or not os.getenv('KRAKEN_API_SECRET'):
        print("⚠️  No API credentials - add to .env file:")
        print("KRAKEN_API_KEY=your_key")
        print("KRAKEN_API_SECRET=your_secret")
        return
    
    config = MultiAssetConfig()
    
    print(f"\n📋 CONFIGURATION:")
    print(f"   Total Capital: ${config.total_available_usd}")
    print(f"   BTC Allocation: ${config.get_allocated_capital('BTC')} (70%)")
    print(f"   SOL Allocation: ${config.get_allocated_capital('SOL')} (30%)")
    
    print(f"\n📊 BTC Settings:")
    print(f"   Position Size: ${config.btc.initial_position_usd}")
    print(f"   Profit Levels: 1.5% / 2.5% / 4.0%")
    print(f"   Stop Loss: Disabled (optional)")
    
    print(f"\n⚡ SOL Settings:")
    print(f"   Position Size: ${config.sol.initial_position_usd}")
    print(f"   Profit Levels: 1.0% / 2.0% / 3.5%")
    print(f"   Stop Loss: 3% (mandatory)")
    
    confirm = input(f"\nStart multi-asset trading? (type 'YES'): ").strip()
    if confirm != 'YES':
        print("Cancelled")
        return
    
    final_confirm = input(f"FINAL - Trade with ${config.total_available_usd} (type 'TRADE'): ").strip()
    if final_confirm != 'TRADE':
        print("Cancelled")
        return
    
    try:
        bot = MultiAssetTradingBot(config)
        bot.run()
        
    except Exception as e:
        logging.error(f"Bot failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()