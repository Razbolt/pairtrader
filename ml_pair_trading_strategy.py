#!/usr/bin/env python3
"""
ML Pair Trading Strategy: Spread Prediction with Walk-Forward Validation

This script implements a machine learning-based pair trading strategy that
predicts the next day's spread value using an XGBoost regression model.
It employs a robust walk-forward validation methodology for backtesting,
ensuring that the model is periodically retrained to adapt to new market data.

Key Components:
1.  Pair Selection: Uses the Engle-Granger cointegration test to find
    statistically significant pairs.
2.  Feature Engineering: Creates technical analysis features (SMAs, EMA,
    RSI, MACD) from the historical spread series.
3.  XGBoost Regressor: Predicts the t+1 spread value.
4.  Walk-Forward Backtesting:
    - Trains on a fixed window (e.g., 4 years).
    - Predicts on a subsequent period.
    - Retrains the model every 5 trading days to capture evolving dynamics.
5.  Trading Logic: Executes trades based on the predicted change in spread
    compared against a transaction cost threshold.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import xgboost as xgb
import warnings
from datetime import timedelta
import joblib
import sys

warnings.filterwarnings('ignore')

class MLSpreadPredictionStrategy:
    """
    Implements a pair trading strategy using an XGBoost model to predict
    spread values and a walk-forward validation approach for backtesting.
    This version trains a specialized model for each cointegrated pair.
    """
    def __init__(self, training_window_years=4, retrain_interval_days=5, significance_level=0.05, min_pairs=5, max_stocks=100, tc_threshold=0.0028):
        self.training_window = timedelta(days=training_window_years * 365)
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.significance_level = significance_level
        self.min_pairs = min_pairs
        self.max_stocks = max_stocks
        self.tc_threshold = tc_threshold
        self.cointegrated_pairs = []
        self.all_trades = {} # Dictionary to store trades per pair
        self.all_spread_histories = {} # Dictionary to store prediction history

    def load_data(self, data_path):
        """
        Loads and combines formation and trading data into a single continuous series.
        This is necessary for the walk-forward validation approach.
        """
        data_path = Path(data_path)
        formation_file = next(data_path.glob("*_in_sample_formation.csv"), None)
        trading_file = next(data_path.glob("*_out_sample_trading.csv"), None)

        if not formation_file or not trading_file:
            raise FileNotFoundError("Could not find formation or trading CSV files.")

        print("🚀 ML SPREAD PREDICTION STRATEGY")
        print("="*60)
        print(f"📈 Loading Formation Data: {formation_file.name}")
        print(f"💰 Loading Trading Data:   {trading_file.name}")
            
        formation_df = pd.read_csv(formation_file)
        trading_df = pd.read_csv(trading_file)

        # Standardize date column and set as index
        for df in [formation_df, trading_df]:
            date_col = 'period' if 'period' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
        # Combine data for continuous backtest
        self.full_data = pd.concat([formation_df, trading_df]).sort_index()
        self.price_columns = [col for col in self.full_data.columns if col.startswith('p_adjclose_') or col.startswith('P_')]
        
        # Handle commodity data which may not have prefixes
        if not self.price_columns:
            # Assuming remaining columns are prices for commodities
            self.price_columns = [col for col in self.full_data.columns if not col.startswith('v_')]

        self.all_prices = self.full_data[self.price_columns]
        print(f"✅ Combined data from {self.full_data.index.min():%Y-%m-%d} to {self.full_data.index.max():%Y-%m-%d}")
        print(f"✅ Found {len(self.price_columns)} price series.")
        return self

    def find_top_cointegrated_pairs(self, target_pair=None):
        """
        Finds the top N most strongly cointegrated pairs from the dataset.
        If a target_pair is specified, it ensures that pair is included if found.
        """
        print(f"\n🔍 Finding top {self.min_pairs} cointegrated pairs (testing top {self.max_stocks} stocks, sig < {self.significance_level*100}%)")
        
        all_found_pairs = []
        
        stocks_to_test = self.price_columns[:self.max_stocks]
        
        for i, stock1 in enumerate(stocks_to_test):
            for j, stock2 in enumerate(stocks_to_test[i+1:]):
                s1 = self.all_prices[stock1].dropna()
                s2 = self.all_prices[stock2].dropna()
                common_dates = s1.index.intersection(s2.index)

                if len(common_dates) < 252: continue

                s1_clean, s2_clean = s1[common_dates], s2[common_dates]
                
                try:
                    coint_stat, p_value, _ = coint(s1_clean, s2_clean)
                    if p_value < self.significance_level:
                        ols_result = OLS(s1_clean, s2_clean).fit()
                        hedge_ratio = ols_result.params[0]
                        all_found_pairs.append({
                            'stock1': stock1, 'stock2': stock2,
                            'stock1_name': stock1.replace('p_adjclose_', '').replace('P_', ''),
                            'stock2_name': stock2.replace('p_adjclose_', '').replace('P_', ''),
                            'p_value': p_value, 'hedge_ratio': hedge_ratio
                        })
                except Exception:
                    continue
        
        if all_found_pairs:
            # Sort by p-value
            sorted_pairs = sorted(all_found_pairs, key=lambda x: x['p_value'])

            if target_pair:
                stock1_name, stock2_name = target_pair.split('-')
                target_pair_info = None
                # Find the specific pair in the full sorted list
                for pair in sorted_pairs:
                    if (pair['stock1_name'] == stock1_name and pair['stock2_name'] == stock2_name) or \
                       (pair['stock1_name'] == stock2_name and pair['stock2_name'] == stock1_name):
                        target_pair_info = pair
                        break
                
                if target_pair_info:
                    print(f"✅ Found specified pair: {target_pair} (p-value: {target_pair_info['p_value']:.4f})")
                    # Make it the only pair to be backtested
                    self.cointegrated_pairs = [target_pair_info]
                else:
                    print(f"❌ Specified pair {target_pair} not found among cointegrated pairs. Cannot proceed.")
                    self.cointegrated_pairs = []
            else:
                # Otherwise, take the top N pairs
                self.cointegrated_pairs = sorted_pairs[:self.min_pairs]

            if not target_pair:
                print(f"🏆 Found {len(self.cointegrated_pairs)} pairs meeting criteria:")
                for k, pair_info in enumerate(self.cointegrated_pairs):
                    print(f"   {k+1}. {pair_info['stock1_name']}-{pair_info['stock2_name']} (p-value: {pair_info['p_value']:.6f}, β: {pair_info['hedge_ratio']:.4f})")
        else:
            print("❌ No cointegrated pairs found.")
        return self

    def _calculate_features(self, spread_series):
        """Calculates technical analysis features for the spread."""
        features = pd.DataFrame(index=spread_series.index)
        
        # SMAs
        features['SMA5'] = spread_series.rolling(window=5).mean()
        features['SMA10'] = spread_series.rolling(window=10).mean()
        features['SMA15'] = spread_series.rolling(window=15).mean()
        
        # EMA
        features['EMA9'] = spread_series.ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = spread_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = spread_series.ewm(span=12, adjust=False).mean()
        ema26 = spread_series.ewm(span=26, adjust=False).mean()
        features['MACD'] = ema12 - ema26
        
        # Previous day's spread
        features['prev_spread'] = spread_series.shift(1)
        
        return features.dropna()

    def train_xgboost_regressor(self, X_train, y_train, X_eval, y_eval):
        """Trains the XGBoost regression model with early stopping."""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 400,
            'max_depth': 8,
            'learning_rate': 0.05,
            'reg_lambda': 1,
            'gamma': 0.005,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 5
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                   eval_set=[(X_eval, y_eval)],
                   verbose=False)
        return model

    def run_walk_forward_backtest(self, save_results_path=None):
        """
        Executes the walk-forward backtesting for each selected pair,
        training a specialized model for each one.
        """
        if not self.cointegrated_pairs:
            return

        all_daily_pnl_dfs = []

        for pair_info in self.cointegrated_pairs:
            pair_name = f"{pair_info['stock1_name']}-{pair_info['stock2_name']}"
            print(f"\n--- Backtesting Pair: {pair_name} ---")

            s1 = self.all_prices[pair_info['stock1']]
            s2 = self.all_prices[pair_info['stock2']]
            common_dates = s1.index.intersection(s2.index)
            
            spread = s1[common_dates] - pair_info['hedge_ratio'] * s2[common_dates]
            
            features = self._calculate_features(spread)
            target = spread.shift(-1)
            
            aligned_index = features.index.intersection(target.index)
            X = features.loc[aligned_index]
            y = target.loc[aligned_index]

            backtest_start_date = X.index.min() + self.training_window
            backtest_date_range = X.index[X.index >= backtest_start_date]

            if backtest_date_range.empty:
                print(f"   ❌ Backtest failed for this pair: Not enough data for the training window.")
                continue

            pair_trades = []
            spread_history = []
            position = 0
            model = None
            last_retrain_date = backtest_date_range.min() - pd.Timedelta(days=999)

            for current_date in backtest_date_range:
                if (current_date - last_retrain_date) >= self.retrain_interval:
                    train_start, train_end_date = current_date - self.training_window, current_date - pd.Timedelta(days=30)
                    eval_start_date = train_end_date + pd.Timedelta(days=1)
                    X_train, y_train = X.loc[train_start:train_end_date], y.loc[train_start:train_end_date]
                    X_eval, y_eval = X.loc[eval_start_date:current_date], y.loc[eval_start_date:current_date]
                    if not X_train.empty and not y_train.empty and not X_eval.empty and not y_eval.empty:
                        model = self.train_xgboost_regressor(X_train, y_train, X_eval, y_eval)
                        last_retrain_date = current_date
                
                if model:
                    today_features = X.loc[[current_date]]
                    predicted_spread_tomorrow = model.predict(today_features)[0]
                    actual_spread_today = spread.loc[current_date]
                    delta_t1 = predicted_spread_tomorrow - actual_spread_today

                    spread_history.append({
                        'date': current_date,
                        'real_spread': actual_spread_today,
                        'predicted_spread': predicted_spread_tomorrow
                    })

                    if position == 0:
                        if delta_t1 > self.tc_threshold:
                            position, entry_price_s1, entry_price_s2, entry_date = 1, s1.loc[current_date], s2.loc[current_date], current_date
                        elif delta_t1 < -self.tc_threshold:
                            position, entry_price_s1, entry_price_s2, entry_date = -1, s1.loc[current_date], s2.loc[current_date], current_date
                    elif (position == 1 and delta_t1 < 0) or (position == -1 and delta_t1 > 0):
                        exit_price_s1, exit_price_s2 = s1.loc[current_date], s2.loc[current_date]
                        pnl = (exit_price_s1 - entry_price_s1) * position + pair_info['hedge_ratio'] * (entry_price_s2 - exit_price_s2) * position
                        pair_trades.append({'entry_date': entry_date, 'exit_date': current_date, 'pnl': pnl})
                        position = 0
            
            # --- FIX: Store results as DataFrames from the start ---
            self.all_trades[pair_name] = pd.DataFrame(pair_trades)
            self.all_spread_histories[pair_name] = pd.DataFrame(spread_history).set_index('date')
            
            # Create a daily PnL series for this pair for overall analysis
            if not self.all_trades[pair_name].empty:
                pair_trades_df = self.all_trades[pair_name]
                pair_daily_pnl_df = pair_trades_df.set_index('exit_date')['pnl'].resample('D').sum().fillna(0).reset_index()
                pair_daily_pnl_df.rename(columns={'exit_date': 'date'}, inplace=True)
                all_daily_pnl_dfs.append(pair_daily_pnl_df)
        
        # Now print the full summary
        self.print_full_results(all_daily_pnl_dfs, save_results_path)
        return all_daily_pnl_dfs

    def print_full_results(self, all_daily_pnl_dfs, root_output_dir):
        """
        Calculates and prints aggregated portfolio results and per-pair breakdowns.
        """
        print("\n" + "="*70)
        print("📊 AGGREGATED PORTFOLIO RESULTS")
        print("="*70)

        # --- AGGREGATED PORTFOLIO CALCULATION ---
        portfolio_pnl_df = pd.concat(all_daily_pnl_dfs)
        if portfolio_pnl_df.empty:
            portfolio_pnl = pd.Series(dtype=float)
        else:
            portfolio_pnl = portfolio_pnl_df.groupby('date')['pnl'].sum().sort_index()

        portfolio_cum_pnl = portfolio_pnl.cumsum()
        
        portfolio_daily_returns = portfolio_cum_pnl.pct_change().fillna(0) # Using CUMULATIVE P&L to get returns
        
        # Calculate Aggregated Sharpe Ratio
        if portfolio_daily_returns.std() > 0:
            portfolio_sharpe = (portfolio_daily_returns.mean() / portfolio_daily_returns.std()) * np.sqrt(252)
        else:
            portfolio_sharpe = 0

        # Calculate Aggregated Sortino Ratio
        downside_returns_portfolio = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_deviation_portfolio = downside_returns_portfolio.std()
        if downside_deviation_portfolio > 0:
            sortino_ratio_portfolio = (portfolio_daily_returns.mean() / downside_deviation_portfolio) * np.sqrt(252)
        else:
            sortino_ratio_portfolio = np.inf


        total_pnl = portfolio_cum_pnl.iloc[-1] if not portfolio_cum_pnl.empty else 0
        
        # Calculate Drawdown
        max_drawdown = (portfolio_cum_pnl - portfolio_cum_pnl.cummax()).min()

        # Combine all trades for aggregate stats
        full_trades_df = pd.concat(self.all_trades.values())
        total_trades = len(full_trades_df)
        
        if total_trades > 0:
            wins = full_trades_df[full_trades_df['pnl'] > 0]
            losses = full_trades_df[full_trades_df['pnl'] < 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            gross_profit = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            avg_pnl_per_trade = full_trades_df['pnl'].mean()
        else:
            win_rate, profit_factor, avg_pnl_per_trade = 0, 0, 0

        print("PERFORMANCE METRICS")
        print("----------------------------------------------------")
        print(f"Total Portfolio P&L:         $ {total_pnl:10.2f}")
        print(f"Portfolio Sharpe Ratio:        {portfolio_sharpe:10.2f}")
        print(f"Portfolio Sortino Ratio:       {sortino_ratio_portfolio:10.2f}")
        print(f"Max Drawdown:                $ {abs(max_drawdown):10.2f}")
        print(f"Profit Factor:                 {profit_factor:10.2f}")
        print("----------------------------------------------------")

        full_trades_df = pd.concat(self.all_trades.values())
        total_trades = len(full_trades_df)
        
        if total_trades > 0:
            wins = full_trades_df[full_trades_df['pnl'] > 0]
            losses = full_trades_df[full_trades_df['pnl'] < 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            
            gross_profit = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            avg_pnl_per_trade = full_trades_df['pnl'].mean()
        else:
            win_rate, profit_factor, avg_pnl_per_trade = 0, 0, 0

        print(f"Total Trades:                  {total_trades:10d}")
        print(f"Win Rate:                      {win_rate:10.2f}%")
        print(f"Avg P&L per Trade:           $ {avg_pnl_per_trade:10.2f}")
        print("----------------------------------------------------")
        
        print("\n--- PERFORMANCE BY PAIR ---")
        
        # Determine the full date range for the trading period for accurate daily stats
        if not full_trades_df.empty:
            trading_start_date = full_trades_df['entry_date'].min()
            trading_end_date = full_trades_df['exit_date'].max()
            date_range = pd.date_range(start=trading_start_date, end=trading_end_date, freq='D')

        for pair_name, trades_df in self.all_trades.items():
            pair_pnl_total = trades_df['pnl'].sum()
            pair_win_rate = (trades_df['pnl'] > 0).mean() * 100 if not trades_df.empty else 0
            
            pair_sharpe = 0
            pair_sortino = 0

            # Calculate per-pair Sharpe and Sortino based on a full daily P&L series
            if not trades_df.empty and not full_trades_df.empty:
                # Create daily P&L series for the pair, reindexed to the full trading period
                pair_daily_pnl = trades_df.set_index('exit_date')['pnl'].resample('D').sum().reindex(date_range, fill_value=0)
                pair_cum_pnl = pair_daily_pnl.cumsum()
                
                # Calculate returns from the *cumulative* P&L to measure return on equity
                pair_daily_returns = pair_cum_pnl.pct_change().fillna(0).replace([np.inf, -np.inf], 0)

                if pair_daily_returns.std() > 0:
                    pair_sharpe = (pair_daily_returns.mean() / pair_daily_returns.std()) * np.sqrt(252)
                
                downside_returns = pair_daily_returns[pair_daily_returns < 0]
                downside_deviation = downside_returns.std()
                if downside_deviation > 0:
                    pair_sortino = (pair_daily_returns.mean() / downside_deviation) * np.sqrt(252)
                else:
                    pair_sortino = np.inf
            
            print(f"\nPair: {pair_name}")
            print(f"  Trades: {len(trades_df)}, P&L: ${pair_pnl_total:.2f}, Win Rate: {pair_win_rate:.2f}%, Sharpe: {pair_sharpe:.2f}, Sortino: {pair_sortino:.2f}")

    def save_pair_results(self, pair_name, trades_df, model, spread_df, root_output_dir):
        """Saves the results for a single pair to a dedicated folder."""
        pair_output_dir = Path(root_output_dir) / pair_name
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        trades_df.to_csv(pair_output_dir / 'backtest_trades.csv', index=False)
        
        # Save spread history
        if not spread_df.empty:
            spread_df.to_csv(pair_output_dir / 'spread_predictions.csv')
        
        # Save model
        if model:
            joblib.dump(model, pair_output_dir / 'xgboost_model.joblib')

def main():
    """Main function to run the ML backtest."""
    parser = argparse.ArgumentParser(
        description="ML Pair Trading Strategy with Walk-Forward Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('data_path', type=str, help='Path to the directory containing pair trading data.')
    parser.add_argument('--min-pairs', type=int, default=5, help='Number of top cointegrated pairs to backtest (default: 5).')
    parser.add_argument('--max-stocks', type=int, default=100, help='Number of top stocks to search for pairs (default: 100).')
    parser.add_argument('--tc_threshold', type=float, default=0.0028, help='Transaction cost threshold for trading (default: 0.0028).')
    parser.add_argument('--train_years', type=int, default=4, help='Number of years for the rolling training window (default: 4).')
    parser.add_argument('--retrain_days', type=int, default=5, help='Interval in days to retrain the model (default: 5).')
    parser.add_argument('--significance', type=float, default=0.05, help='Cointegration p-value significance level (default: 0.05).')
    parser.add_argument('--pair', type=str, default=None, help='Specify a single pair to run (e.g., AAPL-GOOG)')
    
    args = parser.parse_args()

    # --- Create a unique directory name for the results ---
    data_name = Path(args.data_path).name.replace('_prices_48m12m', '')
    results_dir_name = (
        f"{data_name}_pairs-{args.min_pairs}_sig-{args.significance}_"
        f"retrain-{args.retrain_days}_tc-{args.tc_threshold}"
    )
    save_path = Path('backtest_results') / results_dir_name

    try:
        strategy = MLSpreadPredictionStrategy(
            training_window_years=args.train_years,
            retrain_interval_days=args.retrain_days,
            significance_level=args.significance,
            min_pairs=args.min_pairs,
            max_stocks=args.max_stocks,
            tc_threshold=args.tc_threshold
        )

        strategy.load_data(args.data_path)
        strategy.find_top_cointegrated_pairs(target_pair=args.pair)
        
        if not strategy.cointegrated_pairs:
            print("No pairs to backtest. Exiting.")
            return

        all_daily_pnl_dfs = strategy.run_walk_forward_backtest(save_results_path=save_path)
        
        # Print and save full results
        if all_daily_pnl_dfs:
            strategy.print_full_results(all_daily_pnl_dfs, save_path)
            # --- Save Individual Pair Results ---
            for pair_name, trades_list in strategy.all_trades.items():
                trades_df = pd.DataFrame(trades_list)
                
                # Get the feature importance from the last trained model for this pair
                model_to_save = strategy.all_spread_histories[pair_name].get('model', None)
                
                # Save results
                if save_path:
                    spread_history_df = strategy.all_spread_histories.get(pair_name, pd.DataFrame())
                    strategy.save_pair_results(pair_name, trades_df, model_to_save, spread_history_df, save_path)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()