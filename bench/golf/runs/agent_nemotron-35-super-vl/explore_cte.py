import sys
sys.path.insert(0, '/home/hyoklee/clio-core.dataverse/build/bin')
import cte_golf as cg
import clio_cte_core_ext as cte

cg.connect()
client = cg.connect()

# Try to get the tag for the HDF5 file and explore its blobs
tag_str = 'golf/P01.zip/Swing01/P01_Swing01_stream_data.hdf5'
t = cte.Tag(tag_str)
print(f'Tag: {tag_str}')
print(f'Contained blobs: {t.GetContainedBlobs()}')
print(f'Num contained blobs: {len(t.GetContainedBlobs())}')

# Try getting blob sizes
for b in t.GetContainedBlobs()[:5]:
    try:
        size = t.GetBlobSize(b)
        print(f'  Blob {b}: size={size}')
    except Exception as e:
        print(f'  Blob {b}: error getting size: {e}')

# Try reading the description blob
try:
    desc = t.GetBlob('description', t.GetBlobSize('description'))
    print(f'description: {desc[:200]}')
except Exception as e:
    print(f'Error reading description: {e}')
