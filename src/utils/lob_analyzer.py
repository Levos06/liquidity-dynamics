import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def reconstruct_lob(file_path, output_dir, limit_timestamps=3000):
    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Ensure sorted by timestamp if not already
    # df = df.sort_values('timestamp') 
    # (Assuming it is sorted or grouped by timestamp naturally)
    
    unique_timestamps = df['timestamp'].unique()
    print(f"Total unique timestamps: {len(unique_timestamps)}")
    
    if len(unique_timestamps) > limit_timestamps:
        process_timestamps = unique_timestamps[:limit_timestamps]
    else:
        process_timestamps = unique_timestamps
        
    print(f"Processing first {len(process_timestamps)} timestamps...")
    
    # Storage for reconstructed book
    # Shape: (Levels, Timestamps) -> We want (80, 3000) arrays eventually.
    # We will store (3000, 80) first and then transpose if needed.
    # Actually user asks for "two arrays of prices and volumes 80 on 3000 each".
    # So 80 rows, 3000 columns? "80 on 3000 each". Usually (Features, Samples).
    # Let's target (80, 3000).
    
    aggregated_prices = []
    aggregated_volumes = []
    
    # Current Book State
    # bid: price -> amount
    # ask: price -> amount
    bids = {}
    asks = {}
    
    # To speed up, we can group by timestamp
    # But filtering df by timestamp list might be slow if large.
    # Better to iterate through the dataframe.
    
    # Let's use an iterator
    # We only care about the first N timestamps.
    # Identifying the cutoff:
    cutoff_time = process_timestamps[-1]
    
    # Filter DF to only relevant data to speed up iteration
    # mask = df['timestamp'] <= cutoff_time
    # sub_df = df.loc[mask]
    
    # Actually, iterating groupby object is cleaner
    grouped = df.groupby('timestamp', sort=False)
    
    count = 0
    
    for ts, group in tqdm(grouped, total=limit_timestamps):
        if count >= limit_timestamps:
            break
            
        # Check if snapshot
        # If ANY row in group is snapshot, treating it as snapshot update?
        # Usually snapshots are full dumps.
        is_snapshot = (group['action'] == 'snapshot').any()
        
        if is_snapshot:
            # Clear books
            # But wait, usually a snapshot contains ALL levels.
            # If it's a 'partial' snapshot, maybe not.
            # Assuming 'snapshot' action means full book state provided.
            bids = {}
            asks = {}
            
            # Apply all rows
            for _, row in group.iterrows():
                side = row['side'] # 'bid' or 'ask'
                price = row['price']
                amount = row['amount']
                
                if side == 'bid':
                    bids[price] = amount
                elif side == 'ask':
                    asks[price] = amount
                    
        else:
            # Update
            for _, row in group.iterrows():
                side = row['side']
                price = row['price']
                amount = row['amount']
                
                if side == 'bid':
                    if amount == 0:
                        if price in bids:
                            del bids[price]
                    else:
                        bids[price] = amount
                elif side == 'ask':
                    if amount == 0:
                        if price in asks:
                            del asks[price]
                    else:
                        asks[price] = amount
                        
        # Extract Top 40 Bids and Top 40 Asks
        # Bids: Highest prices first
        sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:40]
        # Asks: Lowest prices first
        sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:40]
        
        # Pad if fewer than 40
        while len(sorted_bids) < 40:
            sorted_bids.append((0.0, 0.0))
        while len(sorted_asks) < 40:
            sorted_asks.append((0.0, 0.0))
            
        # Structure: 40 Bids then 40 Asks? Or separate? 
        # User says: "filter 40 orders from each side, get two arrays of prices and volumes 80 on 3000 each".
        # 80 levels total.
        # Order: Typically Asks (reversed?) + Bids? Or Bids + Asks?
        # Let's do: Top Asks (ascending price) ... then Top Bids (descending price)?
        # Or standard "market depth" view:
        # Ask 40 ... Ask 1, Bid 1 ... Bid 40.
        # User asks for "two arrays... 80 on 3000".
        # Let's stack them: [Bids (0-39), Asks (0-39)] -> 80 total.
        
        # User request: Center = First levels (Best Bid/Ask), Edges = Last levels.
        # sorted_bids is [Best Bid ... Deepest Bid]
        # sorted_asks is [Best Ask ... Deepest Ask]
        # We want: [Deepest Bid ... Best Bid, Best Ask ... Deepest Ask]
        
        reversed_bids = sorted_bids[::-1]
        
        current_step_prices = [p for p, v in reversed_bids] + [p for p, v in sorted_asks]
        current_step_vols =   [v for p, v in reversed_bids] + [v for p, v in sorted_asks]
        
        aggregated_prices.append(current_step_prices)
        aggregated_volumes.append(current_step_vols)
        
        count += 1
        
    # Convert to numpy
    # List of lists (3000 items, each 80 long) -> (3000, 80)
    prices_array = np.array(aggregated_prices).T # (80, 3000)
    volumes_array = np.array(aggregated_volumes).T # (80, 3000)
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'prices.npy'), prices_array)
    np.save(os.path.join(output_dir, 'volumes.npy'), volumes_array)
    
    print(f"Saved arrays to {output_dir}")
    print(f"Prices shape: {prices_array.shape}")
    print(f"Volumes shape: {volumes_array.shape}")

if __name__ == "__main__":
    reconstruct_lob(
        'BTC-USDT-L2orderbook-400lv-2025-08-28.parquet',
        'processed_data',
        limit_timestamps=3000
    )
