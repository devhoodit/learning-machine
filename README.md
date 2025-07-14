<div align="center">
    <h1>Learning Machine</h1>
    <p>When you learn machine, the machine also learns you.</p>
</div>

Learning Machine, library helps process data, model construction for machine learning. Quickly build and version-control your data processing pipeline using easy-to-read and easy-to-edit config file. Insipred by Detectron.

## Supported features
- Building data processing pipeline (engine)
- Create an engine with a readable config file (YAML)
- Support widely used data processing engines. (e.g. scikit-learn scalers)

# Quick Start
## Create with config file
```yaml
# config.ymal

# preload custom engines from directory
projects:
    - "projects"

data_engine:
    - StringToDatetime:
        col: datetime
    - FillNa:
        fillwith: 10
        cols:
            - age
    - StandardScaler:
        cols:
            - income
```
Parsing config file and create engine
```python
from learning_machine import create_from_config

bundle = create_from_config("config.yaml")
engine = bundle.data_engine
```
read csv with pandas and apply engine
```python
import pandas as pd

data = pd.read_csv("data.csv")
data = engine(data)
```

## Create with code
```python
from learning_machine.engine import SequentialEngine, StringToDatetime, FillNa, StandardScaler

string_to_datetime_engine = StringToDatetime(col="datetime")
fill_na_engine = FillNa(cols=["age"], fillwith=10)
standard_scaler_engine = StandardScaler(cols=["income"])

seq_engine = SequentialEngine([
    string_to_datetime_engine,
    fill_na_engine,
    standard_scaler_engine
])

engine = seq_engine
```



