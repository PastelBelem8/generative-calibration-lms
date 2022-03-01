# (WIP) Entity-Relationship (ER) Diagram

It's hard to keep up-to-date with so many files and results constantly drawn week after week. In this file, we reflect on potential structures that underly and represent the core aspects of the decisions made along the experiments pipeline.

## 1. What components/entities participate in this process?

__Matrix__: metadata concerning the storage of the different subsets.

Additional considerations when storing the matrices: these should be ordered by name.

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| matrix_uuid | str | unique identifier, created based on filepath, matrix_type, features, (if possible hash of the file) |
| matrix_type | str | matrix purpose (e.g., train, validation, or test) |
| filepath | str | full filepath of the file with the matrix |
| storage_type | str | storage type of the matrix (e.g., local, hdfs, aws) |
| read_classpath | str | classpath of the method to read the matrix |
| read_parameters | dict | hyperparameters used to read the file (e.g., compressed files, fancy header or indexing processing) |
| num_rows | int | number of rows/instances/examples in the dataset |
| num_cols | int | number of columns in the dataset |
| name_cols | List[str] | ordered list of the columns |
| target_cols | List[str] | list of columns used as target columns |
| id_cols | List[str] | list of columns used as unique identifiers of the examples |

__Split__: composite metadata structure, builds on top of `Matrix`. Represents a set of matrices built with the purpose of being evaluated together.

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| split_uuid | str | unique identifier, created based on constituent matrices unique identifiers (`matrix_uuid`). |
| train_matrix_uuid | str | foreign key to training matrix |
| test_matrix_uuid | str | foreign key to test matrix |
| validation_matrix_uuid | Optional[str] | foreign key to validation matrix (note: in some cases we may just use a simple two-way holdout split) |


__Model config__:


| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| model_config_uuid | str | unique identifier, created based on constituent matrices unique identifiers (`matrix_uuid`). |
| model_classpath | str | model classpath (e.g., `sklearn.trees.DecisionTreeClassifier`) |
| model_hyperparameters | Dict[str, Any] | the set of hyperparameters of this model |


__Model__:

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| model_uuid | str | unique identifier, created based on `model_config_uuid` and `matrix_uuid` where the model was trained on. |
| model_config_uuid | str | unique identifier of the model config this model originated from |
| matrix_uuid | str | unique identifier of the matrix the model was trained on |
| model_filepath | str | the full filepath with the model's pickle |


__Predictions__:

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| predictions_uuid | str | unique identifier, hash of the predictions |
| predictions_filepath | str | filepath of the file with the predictions |
| model_uuid | str | unique identifier of the model that originated these predictions |
| matrix_uuid | str | unique identifier of the matrix concerning these predictions |
| columns | List[str] | list of predictions |



__Evaluations__:

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| eval_uuid | str | unique identifier |


__Dataset__:

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| dataset_uuid | str | unique identifier representing the dataset. |
| dataset_type | str | type of the dataset (e.g., question-answering, qa-ex) |
| features | List[str] | list of the features that comprise this dataset |
| target | List[str] | list of the target values of the dataset |
| preprocessing_classpath | str | classpath of the preprocessor of the dataset |
| preprocessing_hyperparameters | List[str] | |


__Experiment__:

| Attribute | Type | Description      |
| --------- | ---- | ---------------- |
| experiment_uuid | str | unique identifier, created based on the hash with the configurations for the experiment. Ideally it should contain some version of the code used to generate it |
| experiment_configs | Dict[str, Any] | all configurations (except the user specific ones) that characterize one experiment |
| task | str | task type (e.g., regression, classification, calibration) |
| description | str | description of the purpose of this experiment (e.g., evaluate calibrators in ex QA datasets) |


## QUESTIONS TO REFLECT
- Questions: how does this schema handle the calibration models?
- Encodings:
- Tokenizers ?
