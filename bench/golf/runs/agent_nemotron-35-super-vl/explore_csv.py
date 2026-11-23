import sys
sys.path.insert(0, '/home/hyoklee/clio-core.dataverse/build/bin')
import cte_golf as cg

cg.connect()

# Read Participant Metadata
meta = cg.read_csv('golf/Documentation.zip/Participant Metadata.csv')
print('=== Participant Metadata ===')
print(meta.to_string())
print()
print('Columns:', list(meta.columns))
print('Shape:', meta.shape)
