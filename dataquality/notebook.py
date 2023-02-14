from datetime import datetime
from functools import partial
from typing import Any, Optional

TRAIN_CSV = "training.csv"
TEST_CSV = "test.csv"


def _run_button_on_clicked(
    _: Any,
    project_name: str,
    run_name: str,
    train_filename: str,
    test_filename: str,
    newsgroups_demo: bool = False,
) -> None:
    import pandas as pd

    import dataquality as dq

    if newsgroups_demo:
        from sklearn.datasets import fetch_20newsgroups

        newsgroups_train = fetch_20newsgroups(subset="train")
        newsgroups_test = fetch_20newsgroups(subset="test")
        labels = newsgroups_train.target_names
        df_train = pd.DataFrame(
            {
                "text": newsgroups_train.data,
                "label": newsgroups_train.target,
            }
        )
        df_test = pd.DataFrame(
            {
                "text": newsgroups_test.data,
                "label": newsgroups_test.target,
            }
        )
    else:
        df_train = pd.read_csv(train_filename)
        labels = list(set(df_train.label))
        try:
            df_test = pd.read_csv(test_filename)
            dq.auto(
                train_data=df_train,
                test_data=df_test,
                labels=labels,
                project_name=project_name,
                run_name=run_name,
            )
        except Exception as e:
            print(f"Could not read test data: {e}")
            dq.auto(
                train_data=df_train,
                labels=labels,
                project_name=project_name,
                run_name=run_name,
            )


def auto_notebook(
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    train_filename: Optional[str] = None,
    test_filename: Optional[str] = None,
    newsgroups_demo: bool = False,
) -> None:
    """auto_notebook

    auto_notebook creates a button in a jupyter notebook environment that
    runs the dataquality.auto function. This is useful for quickly running
    dataquality on a dataset without having to write much code.

    The only prerequisites is that 1) the dataquality package is installed and
    2) you have run dataquality.login() to authenticate.

    Args:
        project_name (Optional[str], optional): The project name. Automatically set
            to "auto_notebook_project" if not supplied.
        run_name (Optional[str], optional): _description_. Automatically set to
            "run_{datetime}" if not supplied.
        train_filename (Optional[str], optional): The filename of the training data.
            Structured as a csv with two columns, "text" and "label". Automatically
            set to "training.csv" if not supplied.
        test_filename (Optional[str], optional): The filename of the test data.
            Structured as a csv with two columns, "text" and "label". Automatically
            set to "test.csv" if not supplied.
        newsgroups_demo (bool, optional): Run a demo with the newsgroups dataset.
            Defaults to False.
    """
    import ipywidgets
    from IPython.display import display

    if train_filename is None:
        train_filename = TRAIN_CSV
    if test_filename is None:
        test_filename = TEST_CSV
    if project_name is None:
        project_name = "auto_notebook_project"
    if run_name is None:
        _datetime = datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S",
        )
        run_name = f"run_{_datetime}"

    run_button = ipywidgets.Button(
        description="Run Galileo 🔭",
    )
    out = ipywidgets.Output()

    # linking button and function together using a button's method
    run_button.on_click(
        partial(
            _run_button_on_clicked,
            project_name=project_name,
            run_name=run_name,
            train_filename=train_filename,
            test_filename=test_filename,
            newsgroups_demo=newsgroups_demo,
        )
    )

    # displaying button and its output together
    display(
        ipywidgets.VBox(
            [
                run_button,
                out,
            ]
        )
    )
