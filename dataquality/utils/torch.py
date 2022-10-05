from typing import Any, Callable, List

def remove_id_collate_fn_wrapper(
    collate_fn: Any, store: Any
) -> Callable:
    """Removes the id from each row and pass the cleaned version on.
    Will be used as a wrapper for the collate function.
    :param collate_fn: The collate function to wrap
    :param store: The store to use to save the ids (currently passed by reference)
    :return: The wrapped collate function
    """

    def remove_id(rows: List[dict]) -> List[dict]:
        """Removes the id from each row and pass the cleaned version on.
        :param rows: The rows to clean
        :return: The cleaned rows
        """
        # Remove id by reference
        ids = []
        clean_rows = []
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if key == "id":
                    ids.append(value)
                elif key:
                    clean_row[key] = value
            clean_rows.append(clean_row)
        store["ids"] = ids
        return collate_fn(clean_rows)

    return remove_id