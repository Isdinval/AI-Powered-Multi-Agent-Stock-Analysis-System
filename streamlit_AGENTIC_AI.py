# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:06:55 2025

@author: Olivi
"""

# ############################################
# I. IMPORTS
# ############################################

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import time
from typing import Dict, Any, Optional

from openai import OpenAI
from plotly.subplots import make_subplots
import streamlit as st


# ############################################
# II. GATHER INFORMATIONS (TECHNICAL, FUNDAMENTAL, NEWS SENTIMETN ANALYSIS)
# ############################################
def fetch_technical_indicators(ticker: str, timeframe: str = "1d", indicator_params: dict = None):
    # Set default parameters if none provided
    if indicator_params is None:
        indicator_params = {
            'macd_short_period': 12,
            'macd_long_period': 26,
            'macd_signal_period': 9,
            'bb_length': 20,
            'bb_std': 2.0,
            'stochastic_k': 14,
            'stochastic_d': 3,
            'atr_length': 14,
            'willr_length': 14,
            'cmf_length': 20,
            'cci_length': 20,
            'mfi_length': 14,
            'adx_length': 14,
            'supertrend_length': 10,
            'supertrend_multiplier': 3.0
        }
        
    # Unpack the parameters
    macd_short_period = indicator_params['macd_short_period']
    macd_long_period = indicator_params['macd_long_period']
    macd_signal_period = indicator_params['macd_signal_period']
    bb_length = indicator_params['bb_length']
    bb_std = indicator_params['bb_std']
    stochastic_k = indicator_params['stochastic_k']
    stochastic_d = indicator_params['stochastic_d']
    atr_length = indicator_params['atr_length']
    willr_length = indicator_params['willr_length']
    cmf_length = indicator_params['cmf_length']
    cci_length = indicator_params['cci_length']
    mfi_length = indicator_params['mfi_length']
    adx_length = indicator_params['adx_length']
    supertrend_length = indicator_params['supertrend_length']
    supertrend_multiplier = indicator_params['supertrend_multiplier']

    data = yf.download(ticker, period="max", interval=timeframe, group_by='Ticker')

    # Transform the DataFrame: stack the ticker symbols to create a multi-index (Date, Ticker), then reset the 'Ticker' level to turn it into a column
    data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)



    # Calculate Moving Average (MA)
    data['SMA_20'] = ta.sma(data['Close'], length=20)
    

    # Calculate Relative Strength Index (RSI)
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # Calculate Exponential Moving Average (EMA)
    data['EMA_50'] = ta.ema(data['Close'], length=50)

    # Calculate Moving Average Convergence Divergence (MACD) dynamically based on periods
    macd = ta.macd(data['Close'], fast=macd_short_period, slow=macd_long_period, signal=macd_signal_period)
    macd_column = f"MACD_{macd_short_period}_{macd_long_period}_{macd_signal_period}"
    macd_signal_column = f"MACDs_{macd_short_period}_{macd_long_period}_{macd_signal_period}"
    data['MACD'] = macd[macd_column]
    data['MACD_Signal'] = macd[macd_signal_column]

    # Calculate Bollinger Bands dynamically based on period and standard deviation
    bb = ta.bbands(data['Close'], length=bb_length, std=bb_std)
    bb_upper_column = f"BBU_{bb_length}_{bb_std}"
    bb_middle_column = f"BBM_{bb_length}_{bb_std}"
    bb_lower_column = f"BBL_{bb_length}_{bb_std}"
    data['BB_Upper'] = bb[bb_upper_column]
    data['BB_Middle'] = bb[bb_middle_column]
    data['BB_Lower'] = bb[bb_lower_column]

    # Average True Range (ATR)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=atr_length)

    # Stochastic Oscillator (STOCH)
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=stochastic_k, d=stochastic_d)
    stoch_k_column = f"STOCHk_{stochastic_k}_{stochastic_d}_{stochastic_d}"
    stoch_d_column = f"STOCHd_{stochastic_k}_{stochastic_d}_{stochastic_d}"
    data['STOCH_K'] = stoch[stoch_k_column]
    data['STOCH_D'] = stoch[stoch_d_column]

    # Moving Average Convergence Divergence Histogram (MACD Histogram)
    data['MACD_Hist'] = macd[macd_column] - macd[macd_signal_column]

    # On-Balance Volume (OBV)
    data['OBV'] = ta.obv(data['Close'], data['Volume'])

    # Williams %R (WILLR)
    data['WILLR'] = ta.willr(data['High'], data['Low'], data['Close'], length=willr_length)

    # Chaikin Money Flow (CMF)
    data['CMF'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=cmf_length)

    # Commodity Channel Index (CCI)
    data['CCI'] = ta.cci(data['High'], data['Low'], data['Close'], length=cci_length)

    # Money Flow Index (MFI)
    data['MFI'] = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=mfi_length)

    # ADX (Average Directional Index)
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=adx_length)
    adx_prefix = f"ADX_{adx_length}"
    data[f"{adx_prefix}"] = adx[f"ADX_{adx_length}"]
    data[f"{adx_prefix}_DMP"] = adx[f"DMP_{adx_length}"]
    data[f"{adx_prefix}_DMN"] = adx[f"DMN_{adx_length}"]

    # Supertrend
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'], length=supertrend_length, multiplier=supertrend_multiplier)
    supertrend_prefix = f"SUPERT_{supertrend_length}_{supertrend_multiplier}"
    data[f"{supertrend_prefix}"] = supertrend[f"SUPERT_{supertrend_length}_{supertrend_multiplier}"]
    data[f"{supertrend_prefix}d"] = supertrend[f"SUPERTd_{supertrend_length}_{supertrend_multiplier}"]
    data[f"{supertrend_prefix}l"] = supertrend[f"SUPERTl_{supertrend_length}_{supertrend_multiplier}"]
    data[f"{supertrend_prefix}s"] = supertrend[f"SUPERTs_{supertrend_length}_{supertrend_multiplier}"]

    # View the DataFrame with technical indicators
    print(data.tail(3))  # Display last 10 rows for review
    return data

def fetch_fundamental_data(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Define the key metrics and their descriptions
    fundamentals = {
        'P/E Ratio': {
            'value': info.get('trailingPE', 'N/A'),
            'description': 'The Price-to-Earnings ratio compares the company\'s stock price to its earnings per share (EPS). A higher ratio may indicate overvaluation or expected growth.'
        },
        'EPS (TTM)': {
            'value': info.get('trailingEps', 'N/A'),
            'description': 'Earnings Per Share (TTM) indicates the portion of a company’s profit allocated to each outstanding share of common stock over the last twelve months.'
        },
        'Market Cap': {
            'value': info.get('marketCap', 'N/A'),
            'description': 'Market Capitalization is the total market value of a company’s outstanding shares, calculated by multiplying stock price by shares outstanding.'
        },
        'Dividend Yield': {
            'value': info.get('dividendYield', 'N/A'),
            'description': 'Dividend Yield shows how much a company returns to its shareholders in the form of dividends relative to its stock price.'
        },
        'Beta': {
            'value': info.get('beta', 'N/A'),
            'description': 'Beta measures a stock\'s volatility in relation to the overall market. A beta >1 suggests higher volatility than the market.'
        },
        'Price/Sales Ratio': {
            'value': info.get('priceToSalesTrailing12Months', 'N/A'),
            'description': 'Price/Sales ratio compares a company\'s stock price to its revenue per share, reflecting valuation relative to sales.'
        },
        'Price/Book Ratio': {
            'value': info.get('priceToBook', 'N/A'),
            'description': 'Price/Book ratio compares a company\'s market value to its book value. A higher ratio suggests higher valuation relative to assets.'
        },
        'Debt/Equity Ratio': {
            'value': info.get('debtToEquity', 'N/A'),
            'description': 'Debt/Equity ratio shows how much debt a company has compared to equity. A higher ratio indicates more risk due to debt reliance.'
        },
        'Current Ratio': {
            'value': info.get('currentRatio', 'N/A'),
            'description': 'Current Ratio is a liquidity measure that shows a company’s ability to pay short-term liabilities with its current assets.'
        },
        'Quick Ratio': {
            'value': info.get('quickRatio', 'N/A'),
            'description': 'Quick Ratio is similar to the Current Ratio but excludes inventory, focusing on the most liquid assets to cover short-term obligations.'
        },
        'Return on Assets (ROA)': {
            'value': info.get('returnOnAssets', 'N/A'),
            'description': 'Return on Assets (ROA) measures how efficiently a company uses its assets to generate profit.'
        },
        'Return on Equity (ROE)': {
            'value': info.get('returnOnEquity', 'N/A'),
            'description': 'Return on Equity (ROE) shows how effectively a company uses shareholders\' equity to generate profit.'
        },
        'Gross Profit': {
            'value': info.get('grossProfits', 'N/A'),
            'description': 'Gross Profit is the difference between revenue and the cost of goods sold, reflecting the efficiency of production.'
        },
        'Operating Margin': {
            'value': info.get('operatingMargins', 'N/A'),
            'description': 'Operating Margin represents the percentage of revenue remaining after covering operating expenses, showing operational efficiency.'
        },
        'Net Profit Margin': {
            'value': info.get('profitMargins', 'N/A'),
            'description': 'Net Profit Margin is the percentage of revenue left after all expenses, taxes, and costs. It reflects overall profitability.'
        },
        'Revenue (TTM)': {
            'value': info.get('totalRevenue', 'N/A'),
            'description': 'Revenue (TTM) is the total amount of income generated by the sale of goods or services in the last twelve months.'
        },
        'Free Cash Flow': {
            'value': info.get('freeCashflow', 'N/A'),
            'description': 'Free Cash Flow is the cash a company generates after capital expenditures, available for debt repayment, dividends, or reinvestment.'
        },
        'Enterprise Value': {
            'value': info.get('enterpriseValue', 'N/A'),
            'description': 'Enterprise Value is the total value of a company, including market capitalization, debt, and cash.'
        },
        'Operating Income': {
            'value': info.get('operatingIncome', 'N/A'),
            'description': 'Operating Income is the profit from normal business operations, excluding other income like interest and taxes.'
        },
        'Total Debt': {
            'value': info.get('totalDebt', 'N/A'),
            'description': 'Total Debt is the sum of all outstanding debt obligations of the company.'
        },
        'Cash': {
            'value': info.get('totalCash', 'N/A'),
            'description': 'Cash is the total amount of liquid cash and equivalents held by the company.'
        },
        'Net Debt': {
            'value': info.get('netDebt', 'N/A'),
            'description': 'Net Debt is total debt minus cash, showing the actual debt load after accounting for liquid assets.'
        },
        'Intangible Assets': {
            'value': info.get('intangibleAssets', 'N/A'),
            'description': 'Intangible Assets include non-physical assets like patents, trademarks, and goodwill.'
        },
        'Capital Expenditures': {
            'value': info.get('capitalExpenditures', 'N/A'),
            'description': 'Capital Expenditures are funds used by a company to acquire or upgrade physical assets like equipment or buildings.'
        },
        'Revenue Growth': {
            'value': info.get('revenueGrowth', 'N/A'),
            'description': 'Revenue Growth measures the rate at which a company’s revenue is increasing over a period of time.'
        },
        'Earnings Growth': {
            'value': info.get('earningsGrowth', 'N/A'),
            'description': 'Earnings Growth measures the rate at which a company’s earnings are increasing over a period of time.'
        },
        'PEG Ratio': {
            'value': info.get('pegRatio', 'N/A'),
            'description': 'The PEG Ratio is the P/E ratio divided by the earnings growth rate, adjusting for growth. A lower PEG suggests undervaluation relative to growth.'
        },
        'Forward P/E': {
            'value': info.get('forwardPE', 'N/A'),
            'description': 'Forward P/E is the P/E ratio based on forecasted earnings, giving an estimate of the company’s future earnings potential.'
        },
        'Price/Free Cash Flow': {
            'value': info.get('priceToFreeCashflows', 'N/A'),
            'description': 'Price/Free Cash Flow ratio compares the company’s market value to its free cash flow, indicating how much investors are paying for each dollar of free cash flow.'
        },
        'Free Cash Flow Yield': {
            'value': info.get('freeCashFlowYield', 'N/A'),
            'description': 'Free Cash Flow Yield is the free cash flow divided by the company’s market capitalization, indicating the return generated from free cash flow.'
        },
        'Dividend Payout Ratio': {
            'value': info.get('dividendPayoutRatio', 'N/A'),
            'description': 'The Dividend Payout Ratio shows the percentage of earnings paid out as dividends to shareholders.'
        },
        'Dividends Paid': {
            'value': info.get('dividendsPaid', 'N/A'),
            'description': 'Dividends Paid is the total amount of dividends distributed to shareholders during a given period.'
        },
        'Shares Outstanding': {
            'value': info.get('sharesOutstanding', 'N/A'),
            'description': 'Shares Outstanding refers to the total number of shares that a company has issued and that are currently held by shareholders.'
        },
        'Shares Float': {
            'value': info.get('floatShares', 'N/A'),
            'description': 'Shares Float is the number of shares that are available for public trading, excluding restricted shares.'
        },
        'Institutional Ownership': {
            'value': info.get('institutionalOwnership', 'N/A'),
            'description': 'Institutional Ownership is the percentage of a company’s shares held by institutional investors, like mutual funds or pension funds.'
        },
        'Insider Ownership': {
            'value': info.get('insiderOwnership', 'N/A'),
            'description': 'Insider Ownership is the percentage of a company’s shares held by its executives, directors, and other insiders.'
        },
        'Institutional Transactions': {
            'value': info.get('institutionalTransactions', 'N/A'),
            'description': 'Institutional Transactions refer to the percentage of shares bought or sold by institutional investors in a given period.'
        },
        'Insider Transactions': {
            'value': info.get('insiderTransactions', 'N/A'),
            'description': 'Insider Transactions refer to the percentage of shares bought or sold by company insiders in a given period.'
        },
        'Analyst Target Price': {
            'value': info.get('targetMeanPrice', 'N/A'),
            'description': 'Analyst Target Price is the average target price set by analysts for the stock.'
        },
        '52 Week High': {
            'value': info.get('fiftyTwoWeekHigh', 'N/A'),
            'description': 'The 52 Week High is the highest price at which a stock has traded over the past 52 weeks.'
        },
        '52 Week Low': {
            'value': info.get('fiftyTwoWeekLow', 'N/A'),
            'description': 'The 52 Week Low is the lowest price at which a stock has traded over the past 52 weeks.'
        },
    }
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(fundamentals).T
    df.columns = ['Value', 'Description']
    
    return df


# Analyze sentiment using VADER
def analyze_sentiment(text: str) -> str:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Fetch financial news with sentiment analysis
def fetch_financial_news_with_sentiment(ticker: str, count: 50) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    news = stock.get_news(count=count)  # Get the news articles from Yahoo Finance

    analyzer = SentimentIntensityAnalyzer()
    articles = []

    for article in news:
        content = article.get('content', {})
        title = content.get('title', '')
        summary = content.get('summary', '')
        text = f"{title}. {summary}"

        sentiment = analyze_sentiment(text)

        article_data = {
            'title': title,
            'summary': summary,
            'sentiment': sentiment,
            'description': content.get('description', 'N/A'),
            'publisher': content.get('provider', {}).get('displayName', 'N/A'),
            'publish_time': content.get('pubDate', 'N/A'),
            'link': content.get('canonicalUrl', {}).get('url', 'N/A'),
        }

        articles.append(article_data)

    return pd.DataFrame(articles)

# Decide action based on sentiment distribution
def decide_action(news_df: pd.DataFrame) -> tuple[str, str]:
    if news_df.empty:
        return "HOLD", "No recent news available for this ticker, so the recommendation is to HOLD."

    counts = news_df['sentiment'].value_counts()
    pos = counts.get('positive', 0)
    neg = counts.get('negative', 0)
    neu = counts.get('neutral', 0)
    total = len(news_df)

    explanation = f"Out of {total} news articles: {pos} positive, {neg} negative, {neu} neutral. "

    if pos > neg and pos >= 0.6 * total:
        action = "BUY"
        explanation += "The majority of recent news articles show a strong positive sentiment, suggesting investor confidence is high."
    elif neg > pos and neg >= 0.6 * total:
        action = "SELL"
        explanation += "The majority of recent news articles are negative, indicating bearish sentiment and potential downside risk."
    else:
        action = "HOLD"
        explanation += "The sentiment is mixed or neutral, so it's better to wait for clearer signals before making a move."

    return action, explanation

# ############################################
# III. LLM SECTION : API + LLM AGENT queries
# ############################################
class LLMClient:
    """Client for interacting with OpenAI's ChatGPT models"""
    
    def __init__(self, api_key=None):
        # Use OpenAI API from Streamlit secrets
        self.api_key = api_key or st.secrets["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model_name = "gpt-3.5-turbo"  # or "gpt-3.5-turbo" for faster/cheaper option
    
    def query(self, prompt, max_tokens=1024):
        """Send a query to ChatGPT and get response"""
        if not self.client:
            return "Error: OpenAI client not properly initialized"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=1.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying ChatGPT: {e}")
            return "Error in LLM processing."


class Agent:
    """Base agent class with LLM capabilities"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def analyze(self, data):
        """Each agent will implement its own analysis method"""
        raise NotImplementedError


class TechnicalAnalystAgent(Agent):
    """Agent specialized in technical analysis"""
    
    def analyze(self, technical_data, num_days=180):
        # Extract the most recent 'num_days' data points
        recent_data = technical_data.tail(num_days).to_dict('records')
        
        # Prepare a list of indicator dictionaries for multiple days
        indicators = []
        for data in recent_data:
            indicators.append({
                "date": str(data.get('Date')),  # Convert date to string to avoid JSON issues
                "close_price": data.get('Close'),
                "volume": data.get('Volume'),
                "rsi": data.get('RSI'),
                "macd": data.get('MACD'),
                "macd_signal": data.get('MACD_Signal'),
                "sma_20": data.get('SMA_20'),
                "ema_50": data.get('EMA_50'),
                "bb_upper": data.get('BB_Upper'),
                "bb_middle": data.get('BB_Middle'),
                "bb_lower": data.get('BB_Lower'),
                "stoch_k": data.get('STOCH_K'),
                "stoch_d": data.get('STOCH_D'),
                "atr": data.get('ATR'),
                "obv": data.get('OBV'),
                "willr": data.get('WILLR'),
                "cmf": data.get('CMF'),
                "mfi": data.get('MFI')
            })
        
        # Get current price and key support/resistance levels
        latest_data = technical_data.iloc[-1]
        current_price = latest_data.get('Close')
        
        # Create improved prompt for the LLM
        prompt = f"""
        You are a professional technical analyst providing actionable insights on a stock. 
        Analyze these technical indicators for the last {num_days} days:
    
        {json.dumps(indicators[-5:], indent=2)}  # Only showing last 5 days to save space
        
        Current price: ${current_price:.2f}
    
        Create a structured technical analysis report with these sections:
    
        1. **Current Price Trend and Momentum**
           - Identify the primary trend (uptrend, downtrend, or sideways)
           - Analyze momentum indicators (MACD, RSI) to determine strength of the trend
           - Note any divergences between price and momentum indicators
    
        2. **Support and Resistance Levels**
           - Identify 2-3 key support levels below current price
           - Identify 2-3 key resistance levels above current price
           - Mention Bollinger Bands and their implications for volatility
    
        3. **Overbought/Oversold Conditions**
           - Evaluate RSI, Stochastics, and Williams %R
           - Determine if the stock is in overbought (>70) or oversold (<30) territory
           - Note any pending reversal signals
    
        4. **Volume Analysis**
           - Analyze volume patterns in relation to price movements
           - Identify volume confirmation or divergence from price action
           - Evaluate On-Balance Volume (OBV) trend
    
        5. **Chart Patterns or Signals**
           - Identify any completed or forming chart patterns
           - Note any crossovers (moving averages, MACD, etc.)
           - Highlight any significant technical signals
    
        6. **Recommendation with Price Targets**
           - Provide a clear BUY, SELL, or HOLD recommendation based solely on technical analysis
           - For BUY/SELL recommendations, specify:
             * Entry price range
             * Stop Loss level (where the technical thesis would be invalidated)
             * Take Profit target(s)
           - Include a confidence level (High, Medium, Low) and timeframe (Short-term, Medium-term, Long-term)
    
        Format your analysis with clear headings and bullet points. Focus on actionable insights rather than theory.
        """
        
        return self.llm_client.query(prompt)


class FundamentalAnalystAgent(Agent):
    """Agent specialized in fundamental analysis"""
    
    def analyze(self, fundamental_data):
        # Convert DataFrame to a more readable format
        fundamentals = {}
        for index, row in fundamental_data.iterrows():
            fundamentals[index] = {"value": row['Value'], "description": row['Description']}
        
        prompt = f"""
        You are a seasoned equity research analyst providing a fundamental analysis report for a stock.
        Analyze these fundamental metrics:
        
        {json.dumps(fundamentals, indent=2)}
        
        Create a structured fundamental analysis report with these sections:
    
        1. **Valuation Assessment**
           - Analyze key valuation metrics (P/E, P/S, P/B ratios) compared to industry averages and historical ranges
           - Determine if the stock appears overvalued, undervalued, or fairly valued
           - Calculate intrinsic value based on available metrics
           - Provide a valuation score (1-10, where 10 is extremely undervalued)
    
        2. **Financial Health and Stability**
           - Evaluate debt levels, liquidity ratios, and cash position
           - Assess balance sheet strength
           - Analyze cash flow sustainability
           - Provide a financial health score (1-10, where 10 is excellent financial condition)
    
        3. **Growth Potential and Profitability**
           - Analyze revenue growth, earnings growth, and margin trends
           - Evaluate ROE, ROA, and ROIC metrics
           - Assess management effectiveness
           - Provide a growth score (1-10, where 10 is exceptional growth prospects)
    
        4. **Competitive Position**
           - Evaluate market share and competitive advantages
           - Assess industry position and business model strength
           - Identify potential threats and opportunities
           - Provide a competitive position score (1-10, where 10 is dominant market position)
    
        5. **Key Risks and Opportunities**
           - List 3-5 primary risks facing the company
           - Identify 3-5 key opportunities for future growth
           - Provide a risk/reward assessment
    
        6. **Fundamental Recommendation**
           - Provide a clear BUY, SELL, or HOLD recommendation based solely on fundamentals
           - State your conviction level (High, Medium, Low)
           - Specify approximate fair value or price target based on fundamentals
           - Suggest holding period (Short-term, Medium-term, Long-term)
    
        Format your analysis with clear headings, bullet points, and concise language. Be specific with numbers where possible and avoid vague statements.
        """
        
        return self.llm_client.query(prompt)
    
class SentimentAnalystAgent(Agent):
    """Agent specialized in news sentiment analysis"""
    
    def analyze(self, sentiment_data):
        # Get overall sentiment statistics
        counts = sentiment_data['sentiment'].value_counts().to_dict()
        pos = counts.get('positive', 0)
        neg = counts.get('negative', 0)
        neu = counts.get('neutral', 0)
        total = len(sentiment_data)
        
        # Take a sample of news titles, their sentiment, and publication dates
        news_sample = sentiment_data[['title', 'sentiment', 'publish_time', 'summary']].head(20).to_dict('records')
        
        prompt = f"""
        You are a specialized news sentiment analyst for financial markets.
        
        Analyze this sentiment data for a stock:
        
        Overall sentiment distribution:
        - Positive: {pos} articles ({pos/total:.1%})
        - Negative: {neg} articles ({neg/total:.1%})
        - Neutral: {neu} articles ({neu/total:.1%})
        - Total: {total} articles
        
        Recent news sample:
        {json.dumps(news_sample, indent=2)}
        
        Create a structured sentiment analysis report with these sections:
    
        1. **Overall Market Sentiment**
           - Quantify the sentiment distribution and what it indicates
           - Compare positive vs. negative news ratio
           - Assess the overall sentiment climate for this stock
           - Provide a sentiment score (1-10, where 10 is extremely positive)
    
        2. **Key News Themes and Impact**
           - Identify 3-5 recurring themes in recent news
           - Analyze how these themes might impact stock price
           - Note any significant recent developments
           - Highlight any upcoming catalysts mentioned in the news
    
        3. **Sentiment Trend Analysis**
           - Determine if sentiment is improving, deteriorating, or stable
           - Identify any sentiment shifts in recent coverage
           - Note any patterns in sentiment around specific news types
           - Compare current sentiment to any historical context if available
    
        4. **Key Positive and Negative Factors**
           - List the most significant positive factors from recent news
           - List the most concerning negative factors from recent news
           - Weigh the relative importance of positive vs. negative factors
           - Identify any extreme sentiment that might indicate contrarian opportunities
    
        5. **Sentiment-Based Recommendation**
           - Provide a clear BUY, SELL, or HOLD recommendation based solely on sentiment analysis
           - State your conviction level (High, Medium, Low)
           - Estimate potential short-term price impact of current sentiment
           - Note any sentiment-based price targets or support/resistance levels
    
        Format your analysis with clear headings, bullet points, and specific insights rather than generalizations.
        Include any important contrarian indicators or sentiment extremes that might signal reversal points.
        """
        
        return self.llm_client.query(prompt)

class StrategyFormulatorAgent(Agent):
    """Agent that combines all analyses to formulate an investment strategy"""
    
    def analyze(self, technical_analysis, fundamental_analysis, sentiment_analysis, ticker):
        prompt = f"""
        You are a holistic investment strategist integrating technical, fundamental, and sentiment analysis to create a comprehensive trading strategy for {ticker}.
        
        You have the following analyses available:
        
        ### TECHNICAL ANALYSIS:
        {technical_analysis}
        
        ### FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}
        
        ### SENTIMENT ANALYSIS:
        {sentiment_analysis}
        
        Create a comprehensive investment strategy that includes these sections:
    
        1. **Executive Summary**
           - Provide a concise 2-3 sentence summary of your overall recommendation
           - State your final BUY, SELL, or HOLD recommendation with conviction level (Strong, Moderate, Weak)
           - For BUY/SELL, specify current price, entry range, stop loss level, and take profit target
    
        2. **Analysis Integration**
           - Identify key areas where technical, fundamental, and sentiment analyses align
           - Highlight major contradictions between different analysis methods
           - Explain how you're weighing conflicting signals (e.g., strong technicals but weak fundamentals)
           - Determine which factors are most relevant in the current market context
    
        3. **Risk/Reward Assessment**
           - Calculate the risk/reward ratio based on entry point, stop loss, and take profit levels
           - Quantify potential upside and downside scenarios with probabilities
           - Identify specific events or factors that could invalidate your thesis
           - Include both technical and fundamental risk factors
    
        4. **Time Horizon and Position Management**
           - Specify the recommended time horizon (days, weeks, months)
           - Provide guidelines for position sizing based on risk profile
           - Include a specific exit strategy with multiple scenarios
           - Suggest criteria for re-evaluating the position
    
        5. **Entry/Exit Strategy**
           - **Entry Points:**
             * Primary entry: [Specific price or range]
             * Alternative entry: [Based on a specific condition]
             * Entry triggers to watch for
           - **Exit Points:**
             * Stop Loss: [Specific price - must include this]
             * Take Profit: [Specific price(s) - must include this]
             * Trailing stop strategy (if applicable)
    
        6. **Final Recommendation**
           - ACTION: [BUY/SELL/HOLD] with [STRONG/MODERATE/WEAK] conviction
           - ENTRY: [Specific price range]
           - STOP LOSS: [Specific price - maximum acceptable loss level]
           - TAKE PROFIT: [Specific price or multiple targets]
           - TIMEFRAME: [Short/Medium/Long term]
           - POSITION SIZE: [Suggested allocation as percentage of portfolio]
    
        FORMAT REQUIREMENTS:
        - Use clear headings and bullet points
        - Include specific price levels for all recommendations
        - Justify your reasoning with specific data points from the analyses
        - Be decisive and clear in your final recommendation
        - Avoid hedging language that creates ambiguity
        """
        
        return self.llm_client.query(prompt)
    
# ############################################
# IV. Generating REPORT based on everything 
# ############################################
class CoordinatorAgent:
    """Coordinates the entire analysis process"""
    
    def __init__(self):
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Initialize specialist agents
        self.technical_analyst = TechnicalAnalystAgent(self.llm_client)
        self.fundamental_analyst = FundamentalAnalystAgent(self.llm_client)
        self.sentiment_analyst = SentimentAnalystAgent(self.llm_client)
        self.strategy_formulator = StrategyFormulatorAgent(self.llm_client)
    
    def analyze_ticker(self, ticker: str, news_count: int = 50) -> Dict[str, Any]:
        """Coordinate full analysis of a ticker"""
        print(f"🔍 Initiating analysis for {ticker}...")
        
        # Step 1: Gather all data
        print(f"📊 Gathering technical data for {ticker}...")
        technical_data = fetch_technical_indicators(ticker)
        
        print(f"📈 Gathering fundamental data for {ticker}...")
        fundamental_data = fetch_fundamental_data(ticker)
        
        print(f"📰 Gathering news and sentiment data for {ticker}...")
        sentiment_data = fetch_financial_news_with_sentiment(ticker, news_count)
        
        # Step 2: Perform specialized analyses
        print(f"🧮 Technical analyst working...")
        technical_analysis = self.technical_analyst.analyze(technical_data, num_days=30)
        print(technical_analysis)
        print('\n\n')
        
        print(f"🔬 Fundamental analyst working...")
        fundamental_analysis = self.fundamental_analyst.analyze(fundamental_data)
        print(fundamental_analysis)
        print('\n\n')
        
        print(f"📊 Sentiment analyst working...")
        sentiment_analysis = self.sentiment_analyst.analyze(sentiment_data)
        print(sentiment_analysis)
        print('\n\n')
        
        # Step 3: Formulate comprehensive strategy
        print(f"🧠 Strategy formulator integrating all analyses...")
        strategy = self.strategy_formulator.analyze(
            technical_analysis, 
            fundamental_analysis, 
            sentiment_analysis,
            ticker
        )
        
        print(strategy)
        
        # Step 4: Prepare final report
        report = {
            "ticker": ticker,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "technical_analysis": technical_analysis,
            "fundamental_analysis": fundamental_analysis,
            "sentiment_analysis": sentiment_analysis,
            "strategy": strategy,
            "raw_data": {
                "technical": technical_data.tail(20).to_dict(),
                "fundamental": fundamental_data.to_dict(),
                "sentiment": sentiment_data.to_dict()
            }
        }
        
        print(f"✅ Analysis complete for {ticker}!")
        return report
    
    def chat_query(self, prompt: str) -> str:
        """Consistent chat interface query method"""
        try:
            return self.llm_client.query(prompt)
        except Exception as e:
            return f"Error in chat processing: {str(e)}"
        
    def generate_report(self, ticker: str, news_count: int = 50) -> str:
        """Generate a formatted report for the ticker"""
        analysis = self.analyze_ticker(ticker, news_count)
        
        report = f"""
        # Investment Analysis Report: {ticker}
        
        **Generated on:** {analysis['analysis_date']}
        
        ## 1. Executive Summary
        
        {analysis['strategy'][:500]}... [See full strategy section for details]
        
        ## 2. Technical Analysis
        
        {analysis['technical_analysis']}
        
        ## 3. Fundamental Analysis
        
        {analysis['fundamental_analysis']}
        
        ## 4. Sentiment Analysis
        
        {analysis['sentiment_analysis']}
        
        ## 5. Comprehensive Strategy
        
        {analysis['strategy']}
        
        ---
        *This report was generated automatically by the Financial Analysis Agent System*
        """
        
        return report
    
   


import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os
from datetime import datetime, timedelta
from openai import OpenAI

# ############################################
# V. Present REPORT as a STREAMLIT APP
# ############################################

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better typography, color variables, and dark mode compatibility
st.markdown("""
<style>
    /* CSS Variables for easy theming */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #0D47A1;
        --accent-color: #29B6F6;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --danger-color: #F44336;
        --light-bg: #ffffff;
        --dark-bg: #121212;
        --light-text: #333333;
        --dark-text: #e0e0e0;
        --card-shadow: rgba(0,0,0,0.1);
        --card-shadow-dark: rgba(0,0,0,0.4);
        --font-primary: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        --font-secondary: 'Arial', sans-serif;
        --border-radius: 10px;
        --spacing-unit: 8px;
    }

    /* Typography improvements */
    body {
        font-family: var(--font-primary);
        font-size: 16px;
        line-height: 1.6;
    }

    /* Main header styling */
    .main-header {
        font-family: var(--font-primary);
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: calc(var(--spacing-unit) * 1);
        text-shadow: 1px 1px 2px var(--card-shadow);
    }

    /* Sub-header styling */
    .sub-header {
        font-family: var(--font-primary);
        font-size: 1.5rem;
        font-weight: 400;
        color: var(--light-text);
        text-align: center;
        margin-bottom: calc(var(--spacing-unit) * 3);
        opacity: 0.85;
    }

    /* Section headers */
    .section-header {
        font-family: var(--font-primary);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-top: calc(var(--spacing-unit) * 3);
        margin-bottom: calc(var(--spacing-unit) * 2);
        padding-bottom: calc(var(--spacing-unit) * 1);
        border-bottom: 2px solid rgba(30, 136, 229, 0.3);
    }

    /* Metric cards with hover effects */
    .metric-card {
        background-color: var(--light-bg);
        border-left: 4px solid var(--primary-color);
        border-radius: var(--border-radius);
        padding: calc(var(--spacing-unit) * 2);
        margin: calc(var(--spacing-unit) * 1.5) 0;
        box-shadow: 0 4px 8px var(--card-shadow);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px var(--card-shadow);
    }

    /* Tooltip styling */
    .metric-card .tooltip {
        visibility: hidden;
        width: 200px;
        background-color: rgba(50, 50, 50, 0.95);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 100%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    .metric-card:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }

    /* Metric values styling */
    .metric-title {
        font-family: var(--font-primary);
        font-weight: 600;
        font-size: 1rem;
        color: var(--light-text);
    }
    
    .metric-value {
        font-family: var(--font-primary);
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 8px 0;
    }

    /* Recommendation styles */
    .recommendation {
        font-family: var(--font-primary);
        font-size: 1.4rem;
        font-weight: 600;
        margin: calc(var(--spacing-unit) * 2) 0;
        padding: calc(var(--spacing-unit) * 1.5);
        border-radius: var(--border-radius);
        text-align: center;
    }
    
    .buy {
        color: var(--success-color);
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid var(--success-color);
    }
    
    .sell {
        color: var(--danger-color);
        background-color: rgba(244, 67, 54, 0.1);
        border: 1px solid var(--danger-color);
    }
    
    .hold {
        color: var(--warning-color);
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid var(--warning-color);
    }

    /* Report sections */
    .report-section {
        margin-top: calc(var(--spacing-unit) * 3);
        padding: calc(var(--spacing-unit) * 3);
        border-radius: var(--border-radius);
        background-color: var(--light-bg);
        box-shadow: 0 4px 15px var(--card-shadow);
    }

    /* Section divider */
    .section-divider {
        border-top: 4px solid var(--primary-color);
        opacity: 0.3;
        border-radius: 2px;
        margin: calc(var(--spacing-unit) * 4) 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 1.5rem;  /* Increased font size to match subheaders */
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 136, 229, 0.1) !important;
        border-bottom: 2px solid var(--primary-color) !important;
    }
    
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }

    
    /* News item styling */
    .news-item {
        padding: calc(var(--spacing-unit) * 2);
        margin-bottom: calc(var(--spacing-unit) * 2);
        border-radius: var(--border-radius);
        border-left: 5px solid;
        background-color: var(--light-bg);
        box-shadow: 0 2px 8px var(--card-shadow);
    }
    
    .news-item.positive {
        border-left-color: var(--success-color);
    }
    
    .news-item.negative {
        border-left-color: var(--danger-color);
    }
    
    .news-item.neutral {
        border-left-color: var(--warning-color);
    }
    
    .news-item .title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    
    .news-item .meta {
        font-size: 0.85rem;
        color: #777;
        margin-bottom: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        font-weight: 600;
        background-color: var(--primary-color);
        color: white;
        border-radius: var(--border-radius);
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px var(--card-shadow);
    }

    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-shadow: var(--card-shadow-dark);
        }
        
        .sub-header {
            color: var(--dark-text);
        }
        
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .metric-title {
            color: var(--dark-text);
        }
        
        .report-section {
            background-color: rgba(255, 255, 255, 0.03);
        }
        
        .news-item {
            background-color: rgba(255, 255, 255, 0.03);
        }
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Mobile responsiveness */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.2rem;
        }
        
        .section-header {
            font-size: 1.4rem;
        }
        
        .metric-value {
            font-size: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def format_recommendation(recommendation_text):
    """Format recommendation text with appropriate color"""
    if "BUY" in recommendation_text:
        return f'<span class="buy">{recommendation_text}</span>'
    elif "SELL" in recommendation_text:
        return f'<span class="sell">{recommendation_text}</span>'
    else:
        return f'<span class="hold">{recommendation_text}</span>'

def display_key_metrics_card(title, value, description, tooltip=None):
    """Display a metric in a nice card format"""
    tooltip_html = f'title="{tooltip}"' if tooltip else ''
    st.markdown(f"""
    <div class="metric-card" {tooltip_html}>
        <p class="metric-title">{title}</p>
        <p class="metric-value">{value}</p>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def plot_stock_price_with_indicators(data, ticker, days=180):
    """Create an interactive stock price chart with selected technical indicators"""
    # Filter to the most recent days
    data = data.tail(days)
    
    # Create expandable filter section for indicator selection
    with st.expander("📌 Select Technical Indicators to Display", expanded=False):
        # Organize indicators into categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Trend Indicators**")
            show_sma = st.checkbox("SMA (20)", True,
                                 help="Simple Moving Average - Shows average price over 20 days. Price above SMA suggests uptrend, below suggests downtrend.")
            show_ema = st.checkbox("EMA (50)", True,
                                 help="Exponential Moving Average - Gives more weight to recent prices. Crossovers with price can signal trend changes.")
            show_supertrend = st.checkbox("Supertrend", True,
                                        help="Trend-following indicator. Green line = buy signal, Red line = sell signal. Works best in trending markets.")
            show_adx = st.checkbox("ADX", True,
                                 help="Average Directional Index - Measures trend strength (not direction). ADX > 25 = strong trend, < 20 = weak/no trend.")
            
        with col2:
            st.markdown("**Momentum Indicators**")
            show_rsi = st.checkbox("RSI (14)", True,
                                 help="Relative Strength Index - Measures overbought (>70) and oversold (<30) conditions. Divergences can signal reversals.")
            show_macd = st.checkbox("MACD", True,
                                  help="Moving Average Convergence Divergence - Signal line crossovers indicate potential buy/sell signals. Above zero = bullish momentum.")
            show_stoch = st.checkbox("Stochastic", True,
                                    help="Stochastic Oscillator - Shows closing price relative to recent range. >80 = overbought, <20 = oversold.")
            show_willr = st.checkbox("Williams %R", True,
                                   help="Similar to Stochastic but inverted. Values above -20 = overbought, below -80 = oversold.")
            
        with col3:
            st.markdown("**Volatility/Volume**")
            show_bb = st.checkbox("Bollinger Bands", True,
                                help="Price bands based on standard deviations. Price near upper band = overbought, lower band = oversold. Squeeze can precede volatility.")
            show_atr = st.checkbox("ATR", True,
                                 help="Average True Range - Measures volatility. Higher values = more volatility. Useful for setting stop-loss levels.")
            show_obv = st.checkbox("OBV", True,
                                 help="On-Balance Volume - Tracks volume flow. Rising OBV confirms price uptrend, falling OBV suggests weakening trend.")
            show_mfi = st.checkbox("MFI", True,
                                  help="Money Flow Index - Volume-weighted RSI. >80 = overbought, <20 = oversold. Divergences can signal reversals.")
    
    # Create tabs for different chart types
    tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📊 Indicators", "📉 Combined View"])
    
    with tab1:
        # Main price chart with selected overlays
        fig_price = go.Figure()
        
        # Add candlestick chart
        fig_price.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add selected indicators
        if show_sma and 'SMA_20' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['SMA_20'], 
                name='SMA 20', 
                line=dict(color='blue', width=2)
            ))
            
        if show_ema and 'EMA_50' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['EMA_50'], 
                name='EMA 50', 
                line=dict(color='orange', width=2)
            ))
            
        if show_bb and all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['BB_Upper'], 
                name='BB Upper', 
                line=dict(color='purple', width=1, dash='dot')
            ))
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['BB_Middle'], 
                name='BB Middle', 
                line=dict(color='purple', width=1)
            ))
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['BB_Lower'], 
                name='BB Lower', 
                line=dict(color='purple', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,0,128,0.1)'
            ))
            
        if show_supertrend and 'SUPERT_10_3.0' in data.columns:
            fig_price.add_trace(go.Scatter(
                x=data.index, 
                y=data['SUPERT_10_3.0'], 
                name='Supertrend',
                line=dict(color='green', width=2)
            ))
            
        fig_price.update_layout(
            title=f'{ticker} Price Chart with Selected Indicators',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
    
    with tab2:
        # Individual indicator charts
        if show_rsi and 'RSI' in data.columns:
            fig_rsi = px.line(data, x=data.index, y=['RSI'], title=f'{ticker} RSI (14)')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        if show_macd and all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=data.index, 
                y=data['MACD'], 
                name='MACD', 
                line=dict(color='blue', width=2)
            ))
            fig_macd.add_trace(go.Scatter(
                x=data.index, 
                y=data['MACD_Signal'], 
                name='Signal', 
                line=dict(color='orange', width=2)
            ))
            fig_macd.update_layout(
                title=f'{ticker} MACD',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_macd, use_container_width=True)
            
        if show_stoch and all(col in data.columns for col in ['STOCH_K', 'STOCH_D']):
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(
                x=data.index, 
                y=data['STOCH_K'], 
                name='Stochastic %K', 
                line=dict(color='blue', width=2)
            ))
            fig_stoch.add_trace(go.Scatter(
                x=data.index, 
                y=data['STOCH_D'], 
                name='Stochastic %D', 
                line=dict(color='red', width=2)
            ))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
            fig_stoch.update_layout(
                title=f'{ticker} Stochastic Oscillator',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_stoch, use_container_width=True)
            
        if show_obv and 'OBV' in data.columns:
            fig_obv = px.line(data, x=data.index, y=['OBV'], title=f'{ticker} On-Balance Volume')
            fig_obv.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_obv, use_container_width=True)
            
        if show_adx and 'ADX_14' in data.columns:
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(
                x=data.index, 
                y=data['ADX_14'], 
                name='ADX', 
                line=dict(color='black', width=2),
            ))
            fig_adx.add_trace(go.Scatter(
                x=data.index, 
                y=data['ADX_14_DMP'], 
                name='+DI', 
                line=dict(color='green', width=1)
            ))
            fig_adx.add_trace(go.Scatter(
                x=data.index, 
                y=data['ADX_14_DMN'], 
                name='-DI', 
                line=dict(color='red', width=1)
            ))
            fig_adx.update_layout(
                title=f'{ticker} ADX with +/- DI',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_adx, use_container_width=True)
            
        if show_atr and 'ATR' in data.columns:
            fig_atr = px.line(data, x=data.index, y=['ATR'], title=f'{ticker} Average True Range')
            fig_atr.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_atr, use_container_width=True)
            
        if show_willr and 'WILLR' in data.columns:
            fig_willr = px.line(data, x=data.index, y=['WILLR'], title=f'{ticker} Williams %R')
            fig_willr.add_hline(y=-20, line_dash="dash", line_color="red")
            fig_willr.add_hline(y=-80, line_dash="dash", line_color="green")
            fig_willr.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_willr, use_container_width=True)
            
        if show_mfi and 'MFI' in data.columns:
            fig_mfi = px.line(data, x=data.index, y=['MFI'], title=f'{ticker} Money Flow Index')
            fig_mfi.add_hline(y=80, line_dash="dash", line_color="red")
            fig_mfi.add_hline(y=20, line_dash="dash", line_color="green")
            fig_mfi.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_mfi, use_container_width=True)
    
    with tab3:
        # Combined view with subplots
        fig_combined = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
        
        # Price chart
        fig_combined.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add selected overlays to price chart
        if show_sma and 'SMA_20' in data.columns:
            fig_combined.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['SMA_20'], 
                    name='SMA 20',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
        if show_ema and 'EMA_50' in data.columns:
            fig_combined.add_trace(go.Scatter(
                x=data.index, 
                y=data['EMA_50'], 
                name='EMA 50',
                line=dict(color='orange', width=2))
            , row=1, col=1)
            
        # Volume or OBV
        if show_obv and 'OBV' in data.columns:
            fig_combined.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(100,100,100,0.3)'
            ), row=2, col=1)
        else:
            fig_combined.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(100,100,100,0.3)'
            ), row=2, col=1)
            
        # Selected momentum indicator
        if show_rsi and 'RSI' in data.columns:
            fig_combined.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ), row=3, col=1)
            fig_combined.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig_combined.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        elif show_macd and all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig_combined.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue', width=2)
            ), row=3, col=1)
            fig_combined.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                name='Signal',
                line=dict(color='orange', width=2)
            ), row=3, col=1)
            
        fig_combined.update_layout(
            title=f'{ticker} Combined Technical View',
            height=800,
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    return data





def display_technical_key_metrics(data):
    """Display key technical metrics in a row of columns with improved styling and tooltips"""
    # Get the most recent values
    latest_data = data.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi_value = latest_data.get('RSI', 'N/A')
        if rsi_value != 'N/A':
            rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
            tooltip = "RSI (Relative Strength Index) measures the speed and change of price movements. Values over 70 suggest overbought conditions that may lead to a reversal. Values under 30 suggest oversold conditions."
            display_key_metrics_card("RSI (14)", f"{rsi_value:.2f}", f"Status: {rsi_status}", tooltip)
        else:
            display_key_metrics_card("RSI (14)", "N/A", "Relative Strength Index", "Technical indicator showing momentum")
    
    with col2:
        macd = latest_data.get('MACD', 'N/A')
        signal = latest_data.get('MACD_Signal', 'N/A')
        if macd != 'N/A' and signal != 'N/A':
            status = "Bullish" if macd > signal else "Bearish"
            tooltip = "MACD (Moving Average Convergence Divergence) shows the relationship between two moving averages. MACD crossing above signal line is bullish; crossing below is bearish."
            display_key_metrics_card("MACD", f"{macd:.2f}", f"Signal: {signal:.2f} | Status: {status}", tooltip)
        else:
            display_key_metrics_card("MACD", "N/A", "Moving Average Convergence Divergence", "Trend-following momentum indicator")
    
    with col3:
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            sma_20 = latest_data.get('SMA_20', 'N/A')
            sma_50 = latest_data.get('SMA_50', 'N/A')
            if sma_20 != 'N/A' and sma_50 != 'N/A':
                status = "Bullish" if sma_20 > sma_50 else "Bearish"
                tooltip = "Moving Averages smooth out price data. SMA20 above SMA50 indicates a bullish trend; SMA20 below SMA50 suggests a bearish trend."
                display_key_metrics_card("Moving Averages", f"SMA20: {sma_20:.2f}", f"SMA50: {sma_50:.2f} | Trend: {status}", tooltip)
            else:
                display_key_metrics_card("Moving Averages", "N/A", "Simple Moving Averages")
        else:
            display_key_metrics_card("Moving Averages", "N/A", "Simple Moving Averages", "Trend indicators that smooth out price action")
    
    with col4:
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            current_price = latest_data.get('Close', 'N/A')
            bb_upper = latest_data.get('BB_Upper', 'N/A')
            bb_lower = latest_data.get('BB_Lower', 'N/A')
            if current_price != 'N/A' and bb_upper != 'N/A' and bb_lower != 'N/A':
                position = "Upper Band" if current_price >= bb_upper else "Lower Band" if current_price <= bb_lower else "Middle"
                tooltip = "Bollinger Bands show volatility and potential overbought/oversold levels. Price near upper band suggests overbought conditions; price near lower band suggests oversold conditions."
                display_key_metrics_card("Bollinger Bands", f"Price: {current_price:.2f}", f"Position: {position}", tooltip)
            else:
                display_key_metrics_card("Bollinger Bands", "N/A", "Price relative to bands")
        else:
            display_key_metrics_card("Bollinger Bands", "N/A", "Price relative to bands", "Volatility indicator showing standard deviations")

def display_fundamental_metrics(fundamental_data):
    """Display key fundamental metrics in a grid with enhanced styling and tooltips"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pe_ratio = fundamental_data.loc['P/E Ratio', 'Value'] if 'P/E Ratio' in fundamental_data.index else 'N/A'
            tooltip = "P/E Ratio compares the stock price to earnings per share. Lower values may indicate undervalued stocks, while higher values may suggest overvaluation or high growth expectations."
            display_key_metrics_card("P/E Ratio", pe_ratio, "Price to Earnings Ratio", tooltip)
            
        with col2:
            market_cap = fundamental_data.loc['Market Cap', 'Value'] if 'Market Cap' in fundamental_data.index else 'N/A'
            if market_cap != 'N/A' and market_cap is not None:
                try:
                    # Format market cap to billions/millions
                    market_cap_float = float(market_cap)
                    if market_cap_float >= 1e9:
                        market_cap = f"${market_cap_float/1e9:.2f}B"
                    elif market_cap_float >= 1e6:
                        market_cap = f"${market_cap_float/1e6:.2f}M"
                except:
                    pass
            tooltip = "Market Capitalization represents the total market value of a company's outstanding shares. Larger caps typically indicate more established companies with lower volatility."
            display_key_metrics_card("Market Cap", market_cap, "Total Market Value", tooltip)
            
        with col3:
            dividend_yield = fundamental_data.loc['Dividend Yield', 'Value'] if 'Dividend Yield' in fundamental_data.index else 'N/A'
            if dividend_yield != 'N/A' and dividend_yield is not None:
                try:
                    yield_float = float(dividend_yield)
                    dividend_yield = f"{yield_float:.2%}"
                except:
                    pass
            tooltip = "Dividend Yield shows the ratio of annual dividends to share price. Higher yields can indicate value stocks or mature companies, but may also signal potential dividend cuts."
            display_key_metrics_card("Dividend Yield", dividend_yield, "Annual dividend yield", tooltip)
            
        with col4:
            debt_equity = fundamental_data.loc['Debt/Equity Ratio', 'Value'] if 'Debt/Equity Ratio' in fundamental_data.index else 'N/A'
            tooltip = "Debt/Equity Ratio measures financial leverage. Lower ratios indicate less debt and lower risk, while higher ratios may signal higher risk but potentially higher returns."
            display_key_metrics_card("Debt/Equity", debt_equity, "Financial leverage ratio", tooltip)
        
        # Second row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            eps = fundamental_data.loc['EPS (TTM)', 'Value'] if 'EPS (TTM)' in fundamental_data.index else 'N/A'
            tooltip = "Earnings Per Share (TTM) shows the company's profit allocated to each outstanding share. Higher EPS indicates stronger profitability."
            display_key_metrics_card("EPS (TTM)", eps, "Earnings Per Share", tooltip)
            
        with col2:
            beta = fundamental_data.loc['Beta', 'Value'] if 'Beta' in fundamental_data.index else 'N/A'
            tooltip = "Beta measures a stock's volatility compared to the market. Beta>1 indicates more volatility than the market, beta<1 suggests less volatility."
            display_key_metrics_card("Beta", beta, "Stock volatility vs market", tooltip)
            
        with col3:
            roe = fundamental_data.loc['Return on Equity (ROE)', 'Value'] if 'Return on Equity (ROE)' in fundamental_data.index else 'N/A'
            if roe != 'N/A' and roe is not None:
                try:
                    roe_float = float(roe)
                    roe = f"{roe_float:.2%}"
                except:
                    pass
            tooltip = "Return on Equity (ROE) measures a company's profitability relative to shareholders' equity. Higher ROE typically indicates more efficient use of capital."
            display_key_metrics_card("ROE", roe, "Return on Equity", tooltip)
            
        with col4:
            profit_margin = fundamental_data.loc['Net Profit Margin', 'Value'] if 'Net Profit Margin' in fundamental_data.index else 'N/A'
            if profit_margin != 'N/A' and profit_margin is not None:
                try:
                    margin_float = float(profit_margin)
                    profit_margin = f"{margin_float:.2%}"
                except:
                    pass
            tooltip = "Net Profit Margin shows the percentage of revenue that becomes profit. Higher margins indicate more efficient operations and stronger pricing power."
            display_key_metrics_card("Profit Margin", profit_margin, "Net income / Revenue", tooltip)
    except Exception as e:
        st.error(f"Error displaying fundamental metrics: {e}")

def display_sentiment_metrics(sentiment_data):
    """Display sentiment distribution and key sentiment metrics with enhanced styling"""
    try:
        # Calculate sentiment counts
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        positive = sentiment_counts.get('positive', 0)
        negative = sentiment_counts.get('negative', 0)
        neutral = sentiment_counts.get('neutral', 0)
        total = len(sentiment_data)
        
        # Display sentiment metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tooltip = "Positive sentiment may indicate bullish market outlook and potentially rising prices in the short term."
            display_key_metrics_card("Positive Sentiment", f"{positive} ({positive/total:.1%})", 
                                    f"{positive} of {total} articles are positive", tooltip)
        
        with col2:
            tooltip = "Negative sentiment may indicate bearish market outlook and potentially falling prices in the short term."
            display_key_metrics_card("Negative Sentiment", f"{negative} ({negative/total:.1%})", 
                                    f"{negative} of {total} articles are negative", tooltip)
        
        with col3:
            tooltip = "Neutral sentiment indicates balanced market coverage with no strong directional bias."
            display_key_metrics_card("Neutral Sentiment", f"{neutral} ({neutral/total:.1%})", 
                                    f"{neutral} of {total} articles are neutral", tooltip)
        
        # Create sentiment pie chart with improved styling
        labels = ['Positive', 'Negative', 'Neutral']
        values = [positive, negative, neutral]
        colors = ['#4CAF50', '#F44336', '#FFC107']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors,
            textinfo='label+percent',
            insidetextorientation='radial',
            hoverinfo='label+value',
            textfont=dict(size=14),
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        # Add a custom note about sentiment analysis
        st.markdown("""
        <div style="background-color: rgba(41, 182, 246, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #29B6F6;">
            <p style="margin: 0;"><strong>About Sentiment Analysis:</strong> News sentiment can be a leading indicator of price movements. 
            High positive sentiment may indicate potential overbought conditions, while high negative sentiment might signal oversold conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recent news with sentiment - improved styling
        st.markdown("### Recent News with Sentiment")
        
        if not sentiment_data.empty:
            for _, row in sentiment_data.head(5).iterrows():
                sentiment_class = "positive" if row['sentiment'] == 'positive' else "negative" if row['sentiment'] == 'negative' else "neutral"
                
                st.markdown(f"""
                <div class="news-item {sentiment_class}">
                    <div class="title">{row['title']}</div>
                    <div class="meta">{row['publish_time']} | {row['publisher']}</div>
                    <div>{row['summary']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No recent news available")
            
    except Exception as e:
        st.error(f"Error displaying sentiment metrics: {e}")



        
def main():
    st.markdown('<h1 class="main-header">📈 AI-Powered Stock Analysis</h1>', unsafe_allow_html=True)

    st.markdown('<p class="sub-header">Advanced Multi-Agent Investment Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👨‍💻 About the Developer")
    st.markdown(""" 
    **Olivier Raymond** — Data Scientist and Engineer with expertise in GenAI, NLP, and deep learning. With a background in robotics research and engineering, 
    I specialize in developing innovative AI solutions ranging from multi-agent LLM systems to predictive financial models with hybrid LSTM architectures.
    
    My recent projects include participating in AI competitions (ranked in top 37% in AI Mathematical Olympiad), developing RAG chatbots, 
    and creating medical image classification systems (ranked in top 19% in RSNA 2024).
    
    🔗 [LinkedIn](https://www.linkedin.com/in/olivier-raymond/) | 🐙 [GitHub](https://github.com/Isdinval) | 💬 [Interactive Resume](https://ask-olivier-raymond.streamlit.app)
    
    Currently seeking new opportunities in data science, machine learning engineering, and AI research.
    """)

    st.markdown("### 🚀 Project Overview")
    st.markdown("""
    This application represents an advanced implementation of **Agentic AI**, utilizing multiple specialized 
    AI agents working in concert to deliver comprehensive investment analysis:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Technical Agent:** Analyzes price patterns and indicators for short to medium-term timing decisions
        - **Fundamental Agent:** Evaluates company financial health and valuation metrics for long-term investment merit
        - **Sentiment Agent:** Processes news and social sentiment to gauge market psychology
        """)
    with col2:
        st.markdown("""
        - **Strategy Agent:** Synthesizes insights from all specialized agents
        - **Coordinator Agent:** Orchestrates the entire analysis workflow and generates the final report
        """)
    
    # Investment Philosophy section
    st.markdown("### 💡 Investment Philosophy")
    st.markdown("""
    The platform is built on the principle that comprehensive investment decisions require analysis across multiple dimensions:
    """)
    
    # Create the table using Markdown
    st.markdown("""
    | Aspect | Fundamental | Technical | Sentiment |
    |--------|-------------|-----------|-----------|
    | **Answers the question** | "Is this a good company?" | "When should I act?" | "What is the crowd's emotion?" |
    | **Focus** | Business health | Price and volume behavior | Investor psychology |
    | **Time horizon** | Long-term | Short to medium term | Very short to medium term |
    | **Risk addressed** | Value traps, poor businesses | Bad timing, missed moves | Crowd bubbles, emotional traps |
    """)
    
    st.markdown("""
    The ultimate goal is to provide actionable trading strategies with clear **direction (BUY/SELL/HOLD)**, 
    **entry price levels**, **stop-loss guidance**, and **profit targets**.
    """)
    
    st.markdown("---")




    # ############################################
    # SECTION : SIDEBAR - ANALYSIS PARAMETER - Analyze Button - About 
    # ############################################
    with st.sidebar:
        st.header("Analysis Parameters")
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Candle Timeframe",
            options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
            index=8,  # Set the default to '1d' as it was before
            help="Select the timeframe for the price data. Note that intraday data (less than 1 day) is only available for the last 60 days."
        )
        
        st.write(f"You selected a timeframe of: {timeframe}")
        
        news_count = st.slider("Number of News Articles", 10, 50, 100)
    
        # api_key = st.text_input("OpenAI API Key", type="password")
        
        # Indicator customization expander
        with st.expander("Advanced Indicator Settings"):
            st.markdown("### MACD Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                macd_short_period = st.number_input("MACD Short", 5, 50, 12)
            with col2:
                macd_long_period = st.number_input("MACD Long", 10, 100, 26)
            with col3:
                macd_signal_period = st.number_input("MACD Signal", 5, 30, 9)
            
            st.markdown("### Bollinger Bands")
            col1, col2 = st.columns(2)
            with col1:
                bb_length = st.number_input("BB Length", 5, 50, 20)
            with col2:
                bb_std = st.number_input("BB Std Dev", 1.0, 3.0, 2.0, step=0.5)
            
            st.markdown("### Stochastic Oscillator")
            col1, col2 = st.columns(2)
            with col1:
                stochastic_k = st.number_input("Stochastic %K", 5, 30, 14)
            with col2:
                stochastic_d = st.number_input("Stochastic %D", 3, 10, 3)
            
            st.markdown("### Other Indicators")
            col1, col2 = st.columns(2)
            with col1:
                atr_length = st.number_input("ATR Length", 5, 30, 14)
                willr_length = st.number_input("Williams %R Length", 5, 30, 14)
                cmf_length = st.number_input("CMF Length", 5, 30, 20)
                cci_length = st.number_input("CCI Length", 5, 30, 20)
            with col2:
                mfi_length = st.number_input("MFI Length", 5, 30, 14)
                adx_length = st.number_input("ADX Length", 5, 30, 14)
                supertrend_length = st.number_input("Supertrend Length", 5, 20, 10)
                supertrend_multiplier = st.number_input("Supertrend Multiplier", 1.0, 5.0, 3.0, step=0.5)

        # Main content area
        if st.button("Analyze", key="analyze_button", use_container_width=True):
            try:
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Pass all indicator parameters to the technical analysis function
                indicator_params = {
                    'macd_short_period': macd_short_period,
                    'macd_long_period': macd_long_period,
                    'macd_signal_period': macd_signal_period,
                    'bb_length': bb_length,
                    'bb_std': bb_std,
                    'stochastic_k': stochastic_k,
                    'stochastic_d': stochastic_d,
                    'atr_length': atr_length,
                    'willr_length': willr_length,
                    'cmf_length': cmf_length,
                    'cci_length': cci_length,
                    'mfi_length': mfi_length,
                    'adx_length': adx_length,
                    'supertrend_length': supertrend_length,
                    'supertrend_multiplier': supertrend_multiplier
                }
        
                # Fetch data with progress updates
                status_text.text("Fetching stock price data...")
                technical_data = fetch_technical_indicators(ticker, timeframe=timeframe, indicator_params=indicator_params)
                progress_bar.progress(20)
        
                status_text.text("Fetching fundamental data...")
                fundamental_data = fetch_fundamental_data(ticker)
                progress_bar.progress(40)
        
                status_text.text("Analyzing news sentiment...")
                sentiment_data = fetch_financial_news_with_sentiment(ticker, news_count)
                progress_bar.progress(60)
        
                status_text.text("Generating AI analysis...")
                coordinator = CoordinatorAgent()
                report = coordinator.generate_report(ticker, news_count)
                progress_bar.progress(100)
                status_text.empty()
        
                # Save everything into session_state
                st.session_state['technical_data'] = technical_data
                st.session_state['fundamental_data'] = fundamental_data
                st.session_state['sentiment_data'] = sentiment_data
                st.session_state['report'] = report
                st.session_state['ticker'] = ticker
        
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.error("Please check the ticker symbol and try again.")
    
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This app uses:
        - Technical indicators from Yahoo Finance
        - Fundamental data from company reports
        - News sentiment analysis using NLP
        - OpenAI's GPT models for comprehensive analysis
        """)
    
    # If analysis was performed (results saved in session_state)
    if 'technical_data' in st.session_state:
    
        ticker = st.session_state['ticker']
        technical_data = st.session_state['technical_data']
        fundamental_data = st.session_state['fundamental_data']
        sentiment_data = st.session_state['sentiment_data']
        report = st.session_state['report']
    
        # Initialize coordinator if not already in session state
        if 'coordinator' not in st.session_state:
            st.session_state.coordinator = CoordinatorAgent(st.secrets["OPENAI_API_KEY"])
            
        # ############################################
        # SECTION 1 : ANALYSIS RESULTS
        # ############################################
        st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
        tab1, tab2, tab3 = st.tabs(["📈 Technical Analysis", "🧮 Fundamental Analysis", "📰 Sentiment Analysis"])
    
        with tab1:
            plot_stock_price_with_indicators(technical_data, ticker)
            st.markdown("### Key Technical Indicators")
            display_technical_key_metrics(technical_data)
            tech_analysis = report.split("## 2. Technical Analysis")[1].split("## 3. Fundamental Analysis")[0]
            # tech_analysis = clean_llm_markdown(tech_analysis)
            st.markdown(tech_analysis)

            
    
        with tab2:
            st.markdown("### Key Fundamental Metrics")
            display_fundamental_metrics(fundamental_data)
            fund_analysis = report.split("## 3. Fundamental Analysis")[1].split("## 4. Sentiment Analysis")[0]
            # fund_analysis = clean_llm_markdown(fund_analysis)
            st.markdown(fund_analysis)
    
        with tab3:
            st.markdown("### News Sentiment Analysis")
            display_sentiment_metrics(sentiment_data)
            sent_analysis = report.split("## 4. Sentiment Analysis")[1].split("## 5. Comprehensive Strategy")[0]
            # sent_analysis = clean_llm_markdown(sent_analysis)
            st.markdown(sent_analysis)

    
            
            
            
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ############################################
        # SECTION 2 : INVESTMENT STRATEGY
        # ############################################
        st.markdown('<h2 class="section-header">🎯 Investment Strategy</h2>', unsafe_allow_html=True)
        strategy = report.split("## 5. Comprehensive Strategy")[1]
        # strategy = clean_llm_markdown(strategy)
        st.markdown(strategy)


            


if __name__ == "__main__":
    main()
    
