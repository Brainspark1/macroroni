import functools

# 1. Central registry to store external custom methods
_PLUGIN_REGISTRY = {}

# 2. The decorator exposed to external users
def register_method(method_name=None):
    """Decorator to register an external function as a method on MyCoreClass."""
    def decorator(func):
        # Use the function's own name if no custom name is provided
        name = method_name or func.__name__
        _PLUGIN_REGISTRY[name] = func
        return func
    return decorator

# 3. Your core class
class ActionFinder:
    def __init__(self, name):
        self.name = name

    def native_method(self):
        return f"Hello from native method, {self.name}!"
    
    def __get_function_dict(self):
        """Returns a list of all registered external method names."""
        return _PLUGIN_REGISTRY
    

    def __getattr__(self, name):
        """Intercepts missing methods and checks the plugin registry."""
        if name in _PLUGIN_REGISTRY:
            # Bind the external function to this instance as a method
            func = _PLUGIN_REGISTRY[name]
            return functools.partial(func, self)
        
        # Fall back to standard behavior if the attribute doesn't exist
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
