import sys
sys.path.insert(0, '/home/hyoklee/clio-core.dataverse/build/bin')
import cte_golf as cg
import numpy as np

cg.connect()

# Look at one HDF5 file to understand structure
# First, list all unique zip directories in manifest
m = cg.manifest()
zip_dirs = sorted(set(r['path'].split('/')[0] for r in m if r['path'].endswith('.hdf5')))
print('Unique participant zips:', len(zip_dirs))
print('First 5:', zip_dirs[:5])
print('Last 5:', zip_dirs[-5:])

# Read one HDF5 to see what streams are inside
# Need to understand the tag structure - the manifest tag is the HDF5 file tag
# But read_array needs the full tag path including stream
# Let's look at the manifest more carefully for stream info
for r in m:
    if 'Swing01' in r['path'] and r['path'].endswith('.hdf5'):
        print(r)
        break

# Try to read a specific stream - need to figure out tag format
# From cte_golf docstring: golf/<zip>/<SwingNN>/<file>.hdf5/<stream>/<data|time_s>
# Let's try the data tag
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
