# dq.finish

Finishes the current logged run and uploads to the console. By default this will also wait for the run to be fully processed by the Galileo server.

```python
def finish(
  last_epoch: Optional[int] = None, wait: bool = True
) -> Optional[Dict[str, Any]]
```



| Arguments    | Text                                                                                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `last_epoch` | If set, only epochs up to this value will be uploaded/processed. This is inclusive, so setting last\_epoch to 5 would upload epochs 0,1,2,3,4,5                                              |
| `wait`       | If true, after uploading the data, this will wait for the run to be processed by the Galileo server. If false, you can manually wait for the run by calling `dq.wait_for_run()` Default True |

**Examples:**

| Returns | Text                |
| ------- | ------------------- |
| `res`   | Server's response.  |

**Examples:**

```python
import dataquality as dq


dq.init("text_classification")

# ... Model training / logging with dq

# Now all done and ready for upload
dq.finish()
```
