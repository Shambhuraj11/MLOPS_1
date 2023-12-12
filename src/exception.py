import sys

def error_msg_detail(error,error_detail:sys):

    _,_,exe_tb=error_detail.exc_info()

    file_name=exe_tb.tb_frame.f_code.co_filename
    error_msg=f"Error occured in python script name [{file_name}] line number [{exe_tb.tb_lineno}] error message [{str(error)}]"
    return error_msg


class CustomException(Exception): # inheriting class
    def __init__(self, error_message,error_detail:sys) -> None:
        super().__init__(error_message)  # inheriting from exception
        self.error_message=error_msg_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message

