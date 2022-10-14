import os
import traceback


class GalileoException(Exception):
    """
    A class for Galileo Exceptions
    """

    # TODO: Add telemetry


class GalileoWarning(Warning):
    """
    A class for Galileo Warnings
    """

    # TODO: Add telemetry


class LogBatchError(Exception):
    """
    An exception used to indicate an invalid batch of logged model outputs
    """

    # TODO: Add telemetry


class AmplitudeException(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str) -> None:
        self.message = message
        if os.environ.get("DQ_TEMELTRY", None) != "FALSE":
            # Show where the exception was called from
            tb = traceback.extract_stack()
            # Get the second previous line of the traceback
            # reverse tb array:

            for ptb in tb[::-1]:
                # ptb = tb[-3]
                print("tb", ptb.filename)
                print("tb", ptb.line)
                print("tb", ptb.lineno)
                # Get the filename and line number
                filename, line_number, function_name, text = ptb
                # Get the filename without the path
                # Get the line of code
                # Get the exception name
                # Print all the information
                print("txxxb")
                print(filename, line_number, function_name, text)
        super().__init__(self.message)
