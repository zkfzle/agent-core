import sys

# IR userFields key
USER_FIELDS = "userFields"
QUERY = "query"
# IR systemFields key
SYSTEM_FIELDS = "systemFields"

INTERACTION = sys.intern("__interaction__")
# for dynamic interaction raised by nodes

INTERACTIVE_INPUT = sys.intern("__interactive_input__")

INPUTS_KEY = "inputs"
CONFIG_KEY = "config"

END_FRAME = "all streaming outputs finish"

END_NODE_STREAM = "end node stream"

LOOP_ID = "__sys_loop_id"

INDEX = "index"
