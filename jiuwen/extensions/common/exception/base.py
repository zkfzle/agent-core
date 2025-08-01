

MAGIC_CODE = "\t"

class JiuWenException(Exception):
    def __init__(
            self,
            message:str,
            *args,
            **kwargs
    )->None:
        super().__init__(message, args, kwargs)


class JiuWenBaseException(Exception):

    def __init__(self, error_code: int, message:str, node_id : str = None, node_name : str = None,
                 node_type : str = None) -> None:
        super().__init__(error_code, message)
        self._error_code = error_code
        self._message = message
        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type

    def __str__(self):
        return f"[{self._error_code}]{self._message}{MAGIC_CODE}"

    @property
    def error_code(self):
        return self._error_code

    @property
    def message(self):
        return self._message

class InterruptException(JiuWenBaseException):
    def __init__(self, error_code: int, message:str) -> None:
        super().__init__(error_code, message)
        self.__error_code = error_code
        self.__message = message