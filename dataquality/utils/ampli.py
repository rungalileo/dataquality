from enum import Enum


class AmpliMetric(str, Enum):
    dq_import = "dq_import"
    dq_init = "dq_init"
    dq_finish = "dq_finish"
    dq_finished = "dq_finished"
    dq_set_labels = "dq_set_labels"
    dq_log_data = "dq_log_data"
    dq_login = "dq_login"
    dq_import_torch = "dq_import_torch"
    dq_import_tensorflow = "dq_import_tensorflow"
    dq_import_spacy = "dq_import_spacy"
    dq_function_call = "dq_function_call"
    dq_general_exception = "dq_general_exception"
    dq_log_data_exception = "dq_log_data_exception"
    dq_galileo_warning = "dq_galileo_warning"
    dq_log_batch_error = "dq_log_batch_error"
    dq_validation_error = "dq_validation_error"
