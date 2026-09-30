# Using Multiple LLM Clients with Pairadigm

The Pairadigm class now supports using multiple LLM clients in a single instance, allowing you to compare annotations from different models.

## Initialization Options

### Option 1: Single Model (Backward Compatible)
```python
from pairadigm import Pairadigm

# Single model - works exactly as before
pairadigm = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    model_name='gemini-3.8-flash',
    target_concept='political bias',
    cgcot_prompts=prompts
)
```

### Option 2: Multiple Models with Same API Key
```python
# Multiple models from the same provider
pairadigm = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    model_name=['gemini-3.8-flash', 'gemini-3.7-flash'],
    api_key='your-api-key',  # Same key for all models
    target_concept='political bias',
    cgcot_prompts=prompts
)
```

### Option 3: Multiple Models with Different API Keys
```python
# Multiple models from different providers
pairadigm = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    model_name=['gemini-3.8-flash', 'gpt-6-luna', 'claude-sonnet-5-5'],
    api_key=['google-key', 'openai-key', 'anthropic-key'],
    reasoning=['high', None, 'low'],
    target_concept='political bias',
    cgcot_prompts=prompts
)
```

### Option 4: Pre-initialized LLMClients
```python
from pairadigm import LLMClient

# Create custom clients
client1 = LLMClient(
    model_name='gemini-3.8-flash', api_key='key1', reasoning='high'
)
client2 = LLMClient(model_name='gpt-6-luna', api_key='key2')

pairadigm = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    llm_clients=[client1, client2],
    target_concept='political bias',
    cgcot_prompts=prompts
)
```

## Working with Multiple Clients

### View Available Clients
```python
# Get information about all clients
clients_info = pairadigm.get_clients_info()
print(clients_info)
# Output:
#    index              model_name  provider
# 0      0       gemini-3.8-flash    google     high
# 1      1            gpt-6-luna    openai     None
# 2      2  claude-sonnet-5-5  anthropic      low
```

`reasoning` accepts one value for all clients or a list aligned with
`model_name`. A `None` entry leaves that client's provider/model default
unchanged. Named levels are translated for each provider. Google additionally
accepts an integer thinking-token budget, while Anthropic accepts integer
budgets of at least 1024 tokens.

### Generate Breakdowns with Specific Client
```python
# Generate breakdowns using the first client (index 0)
pairadigm.generate_breakdowns_from_paired(client_index=0)

# Generate breakdowns using the second client (index 1)
pairadigm.generate_breakdowns_from_paired(client_index=1)

# Generate breakdowns using the third client (index 2)
pairadigm.generate_breakdowns_from_paired(client_index=2)
```

### Generate Annotations with Multiple Clients
```python
# Annotate with a single client
pairadigm.generate_pairwise_annotations(client_indices=0)

# Annotate with specific clients
pairadigm.generate_pairwise_annotations(client_indices=[0, 1])

# Annotate with all clients (default)
pairadigm.generate_pairwise_annotations()  # Uses all clients
```

When using multiple clients, the DataFrame will contain separate columns for each model:
- `decision_<model_name>` - Decision from each model
- `justification_<model_name>` - Justification from each model
- `breakdown1_<model_name>` and `breakdown2_<model_name>` - Breakdowns from each model

### Score Items from Different Models
```python
# Score using decisions from the first model
scores_model1 = pairadigm.score_items(decision_col='decision_gemini-3.8-flash')

# Score using decisions from the second model
scores_model2 = pairadigm.score_items(decision_col='decision_gpt-6-luna')

# Score using decisions from the third model
scores_model3 = pairadigm.score_items(decision_col='decision_claude-sonnet-5-5')
```

The resulting DataFrames will have score columns named:
- `Bradley_Terry_Score_<model_name>` for each model

## Complete Example Workflow

```python
import pandas as pd
from pairadigm import Pairadigm

# Load your data
df = pd.read_csv('my_data.csv')

# Load CGCoT prompts
prompts = ['prompt1 with {text}', 'prompt2 with {text}']

# Initialize with multiple models
pairadigm = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    paired=True,
    item_id_cols=['item1', 'item2'],
    item_text_cols=['text1', 'text2'],
    model_name=['gemini-3.8-flash', 'gpt-6-luna'],
    target_concept='sentiment',
    cgcot_prompts=prompts
)

# Check available clients
print(pairadigm.get_clients_info())

# Generate breakdowns for each model
pairadigm.generate_breakdowns_from_paired(client_index=0)  # Gemini
pairadigm.generate_breakdowns_from_paired(client_index=1)  # GPT-4

# Generate annotations from both models
pairadigm.generate_pairwise_annotations()  # Uses all clients

# Score items using each model's decisions
scores_gemini = pairadigm.score_items(decision_col='decision_gemini-3.8-flash')
scores_gpt = pairadigm.score_items(decision_col='decision_gpt-6-luna')

# Compare scores between models
comparison = pd.merge(
    scores_gemini[['item_id', 'Bradley_Terry_Score_gemini-3.8-flash']],
    scores_gpt[['item_id', 'Bradley_Terry_Score_gpt-6-luna']],
    on='item_id'
)
print(comparison.corr())
```

## Backward Compatibility

All existing code using a single model will continue to work without any changes. The class automatically:
- Sets `self.client` to the first client for backward compatibility
- Uses default column names (`decision`, `justification`, `breakdown1`, `breakdown2`) when only one client is present
- Defaults to using the first client (index 0) if no client is specified

## Benefits of Multi-Client Support

1. **Model Comparison**: Compare how different LLMs annotate the same items
2. **Ensemble Approaches**: Combine annotations from multiple models for more robust results
3. **Cost-Performance Tradeoffs**: Use different models for different stages (e.g., fast model for initial screening, premium model for final annotations)
4. **Cross-Validation**: Validate results across different model architectures and providers
