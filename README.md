# AI-Powered Multi-Agent Stock Analysis System

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

An advanced investment analysis platform that combines technical indicators, fundamental metrics, and news sentiment analysis through specialized AI agents to generate comprehensive stock recommendations.

## 🌟 Key Features

- **Multi-Agent Architecture**: Four specialized AI agents working in concert:
  - 📈 **Technical Analyst**: Analyzes 20+ indicators (RSI, MACD, Bollinger Bands, etc.)
  - 🧮 **Fundamental Analyst**: Evaluates 30+ financial metrics (P/E, ROE, Debt/Equity, etc.)
  - 📰 **Sentiment Analyst**: Processes news articles with VADER sentiment analysis
  - 🎯 **Strategy Formulator**: Synthesizes insights into actionable recommendations

- **Comprehensive Reporting**:
  - Interactive visualizations of technical indicators
  - Fundamental metrics dashboard
  - News sentiment distribution analysis
  - Clear BUY/SELL/HOLD recommendations with price targets

- **Customizable Analysis**:
  - Adjustable indicator parameters
  - Configurable timeframes (1m to 3mo)
  - Variable news article count

## 🛠️ Technical Stack

- **Backend**: Python 3.9+
- **Data**: Yahoo Finance API (`yfinance`)
- **Technical Analysis**: `pandas_ta` library
- **Sentiment Analysis**: VADER sentiment
- **AI**: OpenAI GPT-3.5/4 for analysis generation
- **Frontend**: Streamlit for interactive web interface
- **Visualization**: Plotly for dynamic charts

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API key
- Streamlit

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-analysis-ai.git
   cd stock-analysis-ai

## 📊 Sample Analysis Workflow
1. Enter Ticker Symbol: (e.g., AAPL, TSLA, MSFT)
2. Select Timeframe: Choose from 1m to monthly charts
3. Adjust Indicators: Customize technical parameters
4. Set News Count: Determine how many articles to analyze
5. Run Analysis: Click "Analyze" button
6. Review Results:
  - Technical charts and indicators
  - Fundamental metrics dashboard
  - News sentiment breakdown
  - AI-generated investment strategy

## 📝 Output Example

The system provides:
- Clear recommendation (BUY/SELL/HOLD)
- Entry price range
- Stop-loss level
- Take-profit targets
- Timeframe guidance
- Position sizing suggestions
