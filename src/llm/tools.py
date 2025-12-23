# ------------------------------------------------------------------
# OpenAI Client Tools for AppWorld
# ------------------------------------------------------------------
TOOLS = [
    {
        "type" : "function",
        "name" : "appworld_execute",
        "description" : """
        Execute Python code inside the AppWorld IPython-like execution shell. 
        The shell is stateful across calls. 
        Use `apis.{app}.{api}(...)` or `requester.{method}(...)` inside the code. 
        When you finish the task, call `apis.supervisor.complete_task()` 
        (optionally with answer=...).
        """,
        "parameters" : {
            "type" : "object",
            "properties" : {
                "code" : {"type" : "string", "description" : "Python code to run in AppWorld shell"}
            },
            "required" : ["code"],
            "additionalProperties" : False
        },
        "strict" : True
    }
]