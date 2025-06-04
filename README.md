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
