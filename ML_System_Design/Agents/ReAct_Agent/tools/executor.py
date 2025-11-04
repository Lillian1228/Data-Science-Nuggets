from typing import Dict, Any, Callable

class ToolExecutor:
    """
    A lightweight tool executor responsible for registering and invoking tools.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable):
        """
        Register a new tool into the toolbox.
        """
        if name in self.tools:
            print(f"Warning: tool '{name}' already exists and will be overwritten.")
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool '{name}' registered.")

    def getTool(self, name: str) -> Callable:
        """
        Retrieve a tool's callable by name.
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        Return a formatted string describing all available tools.
        """
        return "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )
