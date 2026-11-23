import sys
sys.path.insert(0, '/home/hyoklee/clio-core.dataverse/build/bin')
import cte_golf as cg

cg.connect()

# Find all data stream tags for P01 Swing01
tags_data = cg.tags(r'golf/P01\.zip/Swing01/P01_Swing01_stream_data\.hdf5/data')
print(f'Data tags for P01 Swing01: {len(tags_data)}')
for t in tags_data[:5]:
    print(f'  {t}')

# Find all time_s tags
tags_time = cg.tags(r'golf/P01\.zip/Swing01/P01_Swing01_stream_data\.hdf5/time_s')
print(f'Time tags for P01 Swing01: {len(tags_time)}')
for t in tags_time[:5]:
    print(f'  {t}')

# Try read_array on a data tag
try:
    arr = cg.read_array('golf/P01.zip/Swing01/P01_Swing01_stream_data.hdf5/data')
    print(f'data shape: {arr.shape}, dtype: {arr.dtype}')
except Exception as e:
    print(f'Error reading data: {e}')

try:
    arr = cg.read_array('golf/P01.zip/Swing01/P01_Swing01_stream_data.hdf5/time_s')
    print(f'time_s shape: {arr.shape}, dtype: {arr.dtype}')
except Exception as e:
    print(f'Error reading time_s: {e}')