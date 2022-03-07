import json

def convert_json(obj, depth=0):
    """ Convert obj to a version which can be serialized with JSON. """
    if(depth > 10):
        return ""

    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k, depth+1): convert_json(v, depth+1) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x, depth+1) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x, depth+1) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__, depth+1)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k, depth+1): convert_json(v, depth+1) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False