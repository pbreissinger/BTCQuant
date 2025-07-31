# Multi-Asset Trading Bot (BTC + SOL)

A sophisticated cryptocurrency trading bot that simultaneously trades Bitcoin (BTC) and Solana (SOL) using advanced technical analysis and risk management strategies.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Live%20Web%20Interface-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Exchange](https://img.shields.io/badge/Exchange-Kraken-purple)

## 🚀 Features

### **Multi-Asset Strategy**
- **70% BTC / 30% SOL allocation** with separate risk profiles
- **Position stacking** - Dollar-cost averaging on dips
- **Multi-level profit taking** (3 levels per asset)
- **Asset-specific configurations** optimized for volatility

### **Advanced Technical Analysis**
- **Fibonacci retracements** for key support/resistance levels
- **Bollinger Band squeeze detection** for breakout opportunities
- **Multi-timeframe RSI analysis** (7, 14, 21 periods)
- **MACD momentum confirmation**
- **EMA trend alignment** (3, 8, 21 periods)
- **Support/resistance level detection**

### **Risk Management**
- **SOL stop loss** (3% mandatory for high volatility)
- **BTC optional stop loss** (8% if enabled)
- **Daily loss limits** ($25 default)
- **Consecutive loss protection** (max 4 losses)
- **Position size limits** per asset

### **Real-Time Dashboard**
- **Modern financial app UI** (Robinhood/Kraken style)
- **Live price feeds** and position tracking
- **Technical indicator visualization**
- **Performance metrics** and P&L tracking
- **Mobile-responsive design**

## 📊 Trading Strategy

### **BTC Configuration**
- **Position Size**: $25 initial, $20 additional entries
- **Max Positions**: 5 concurrent stacks
- **Profit Levels**: 1.5% / 2.5% / 4.0%
- **Stack Spacing**: 0.8% price drops
- **Stop Loss**: Disabled (optional 8%)

### **SOL Configuration**
- **Position Size**: $15 initial, $12 additional entries
- **Max Positions**: 4 concurrent stacks
- **Profit Levels**: 1.0% / 2.0% / 3.5%
- **Stack Spacing**: 0.7% price drops
- **Stop Loss**: 3% (mandatory)

## 🛠️ Setup Instructions

### **1. Prerequisites**
- Python 3.8 or higher
- Kraken Pro account with API access
- $500+ trading capital (recommended)

### **2. Installation**

```bash
# Clone the repository
git clone <your-repo-url>
cd multi-asset-trading-bot

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
