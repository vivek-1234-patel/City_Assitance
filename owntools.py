from langchain.tools import tool

@tool#decorator for creating a tool that can be used in a chain
def get_greeting(name: str) -> str: #type hint for the input parameter
    """Generate a greeting message for a given name.""" #docstring for the function

    return f"Hello, {name}, welcome to Ai World!" #return statement with a formatted string

result = get_greeting.invoke({"name": "John"}) #invoke the tool with a dictionary containing the input parameter   

print(result) #print the result of the tool invocation

print(get_greeting.name) #print the name of the tool
print(get_greeting.description) #print the description of the tool