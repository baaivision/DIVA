class Registry:
    mapping = {
        "data_name_mapping": {},
    }

    @classmethod
    def data_builder(cls, name):
        def wrap(data):
            cls.mapping["data_name_mapping"][name] = data
            return data
        return wrap
    
    @classmethod
    def get_data_builder(cls, name):
        return cls.mapping["data_name_mapping"].get(name, None)


registry = Registry()
