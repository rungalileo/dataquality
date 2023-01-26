# dq.init

```python
def init(
    task_type: str,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    is_public: bool = True,
) -> None:
```

Initialize a new run and new project, initialize a new run in an existing project, or reinitialize an existing run in an existing project (will overwrite existing run upon `dq.finish`).

Random names are generated if not provided 🐨

| Arguments      | Text                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| `task_type`    | The task type for modeling. This must be one of the valid `dq.schemas.task_type.TaskType` options.         |
| `project_name` | The project name. If not passed in, a random one will be generated.                                        |
| `run_name`     | The run name. If not passed in, a random one will be generated.                                            |
| `is_public`    | Enterprise only. All projects are private in Galileo Cloud. Sets the project's visibility. Default `True`. |

****

**Examples:**

Typically used at the very beginning of your script/module/etc once to set the context for this run.

```python
import dataquality as dq

dq.init(
    task_type="text_classification", 
    project_name="my_awesome_project", 
    run_name="my_awesome_run"
)

# ... Your ML training code and the other dq functions
```
