# Phoenix-II-Dev

## Code Usage

MODEL_PATTERN (directory name under src/model)

> 'phoenix11' means the logic for 1.1 of *Experiments Pipeline And Conclusion*. The rest are by analogy.


0. Copy Basemodel's config.json to new path. *Forbidden to modify the base locally at the original path!*
1. Modify new path Basemodel's Config according addconfig under */src/model/MODEL_PATTERN*
    - [phoenix11](/src/model/phoenix11/addconfig.json), [phoenix11moe](/src/model/phoenix11moe/addconfig.json). 
    - [phoenix12](/src/model/phoenix12/addconfig.json), [phoenix12moe](/src/model/phoenix12moe/addconfig.json). 
2. Reinitialize IModel (unnecessary for phoenix11 MODEL_PATTERN [just copy base model])

    ```
    # Remember to adjust Arguments
    bash ./utils/initModel/iniModel.sh 
    ```

3. Train IModel

    ```
    # Remember to adjust Arguments
    bash train.sh 
    ```


## Code Architechure

- scripts: deepspeed script
- src: source code
    - dataset: Data input and processing logic and dialogue template
    - model: modified modeling.py with suffixes correspond to the experimental setup above
    - trainier.py: Reserve the interface in case modifying the training logic
    - train.py: Main script
- utils: some functional scripts (compress/initialize/download model)
- train.sh: training scripts
