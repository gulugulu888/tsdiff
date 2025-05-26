import json
from pathlib import Path
import pandas as pd

test_file = Path("data/processed_fdr_interactive/CAS_origrate_filtByCAS30p0units/test.jsonl")

print(f"Checking file: {test_file}")
print(f"File exists: {test_file.exists()}")

if test_file.exists():
    print(f"File size: {test_file.stat().st_size} bytes")
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
        print(f"Number of lines: {len(lines)}")
        
        if lines:
            print("\nFirst line sample:")
            first_line = lines[0]
            print(first_line[:500])  # First 500 chars
            
            try:
                first_entry = json.loads(first_line)
                print("\nFirst entry structure:")
                print(f"Keys: {list(first_entry.keys())}")
                print(f"Start field: {first_entry.get('start')}")
                print(f"Target length: {len(first_entry.get('target', []))}")
                print(f"Item ID: {first_entry.get('item_id')}")
                
                # Try to parse the start field
                start_val = first_entry.get('start')
                if start_val:
                    print(f"\nStart field type: {type(start_val)}")
                    print(f"Start field value: {start_val}")
                    
                    # Try to create a Period
                    try:
                        if isinstance(start_val, str) and start_val.startswith("Period"):
                            print("Start field appears to be a Period string representation")
                            # Extract date from Period string
                            import re
                            match = re.search(r"Period\('([^']+)'", start_val)
                            if match:
                                date_str = match.group(1)
                                print(f"Extracted date: {date_str}")
                                ts = pd.Timestamp(date_str)
                                period = pd.Period(ts, freq='250L')
                                print(f"Successfully created Period: {period}")
                        else:
                            ts = pd.Timestamp(start_val)
                            period = pd.Period(ts, freq='250L')
                            print(f"Successfully created Period: {period}")
                    except Exception as e:
                        print(f"Error creating Period: {e}")
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            print("File is empty!")