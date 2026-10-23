# Configurations

YAML files, loaded with `ovannot.core.config.load_config`: `extends:` composes files and
dotted overrides (`key.sub=value`) adjust them from the command line. The pydantic models that
validate each kind are added with the code that uses them.

| Folder | One file per | Holds | From |
|---|---|---|---|
| `models/` | model | `kind` (wrapper), exact checkpoint and revision, licence and where it was checked, endpoint or device, precision, coordinate format and its source | M2 |
| `datasets/` | dataset or subset | loader name, data paths, splits, dev subset | M1 |
| `experiments/` | experiment | pipeline, dataset, setting, overrides of the pipeline's `config.yaml` | M3 |

No secret goes in these files: tokens and keys come from the environment (`HF_TOKEN`,
`ROBOFLOW_API_KEY`).
