# Phoenix-II-Dev


## Running environment. 


## Code Usage

1. we need to download the Qwen model to the current path. For example ./Qwen2.5-1.5B-Instruct. 
2. Modify new path Basemodel's Config according addconfig under */src/model/MODEL_PATTERN*
    - [qwen_decoder](/src/model/qwen_decoder/addconfig.json)
3. Reinitialize IModel (unnecessary for qwen_decoder [MODEL_PATTERN] [just copy base model])
    ```
    # Remember to adjust Arguments
    bash ./utils/initModel/iniModel.sh 
    ```

4. Train IModel

    ```
    # Remember to adjust Arguments
    bash train.sh 
    ```


## Code Architechure

- scripts: deepspeed script
- src: source code
    - dataset: Data input and processing logic and dialogue template
    - model: modified modeling.py with suffixes correspond to the experimental setup above
    - train.py: Main script
- utils: some functional scripts (compress/initialize/download model)
- train.sh: training scripts



## Todo List

- [x] add relevant scores in the forward pass

please refer to the code in the 

src/model/qwen_decoder/modeling.py


```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if kwargs.get("relevant_scores", None) is not None:
        relevant_scores = kwargs["relevant_scores"].unsqueeze(1).unsqueeze(1)
        # normalize relevant scores
        relevant_scores = relevant_scores / relevant_scores.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights * relevant_scores

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

- [ ] change the layer index to make sure only update the final layer. 
- [ ] change the datasets to QA pairs. 
