import pandas as pd
import numpy as np
from pairadigm.scoring import score_items

pairwise_data = {
    'item1': ['A', 'A', 'B', 'C'],
    'item2': ['B', 'C', 'C', 'A'],
    'decision': ['Text1', 'Tie', 'Text2', 'Text1']
}
data = pd.DataFrame({'item': ['A', 'B', 'C']})
pairwise_df = pd.DataFrame(pairwise_data)

res = score_items(
    pairwise_df=pairwise_df,
    data=data,
    item_id_name='item',
    target_concept='test',
    text_name=None,
    paired=False,
    item_id_cols=None,
    use_davidson=True
)
print("\nResults:")
print(res)
