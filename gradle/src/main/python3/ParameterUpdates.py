from addict import Dict


class ParameterUpdates:

    def __init__(self):
        self.__dict__["update_map__"] = Dict({
            ParameterUpdatesAux.default_key: {},
            ParameterUpdatesAux.activities_key: {}
        })

    def __getitem__(self, item):
        return self.update_map__.get(ParameterUpdatesAux.default_key).get(item)

    def __setitem__(self, key, value):
        self.update_map__.get(ParameterUpdatesAux.default_key)[key] = value

    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return self.get(name)
        return self.update_map__.get(ParameterUpdatesAux.default_key).get(name)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("ParameterUpdates object attribute \"{0}\" is read-only".format(name))
        self.update_map__.get(ParameterUpdatesAux.default_key)[name] = value

    def for_activity(self, activity_name):
        activity_dict = self.update_map__.get(ParameterUpdatesAux.activities_key)
        if activity_name not in activity_dict:
            activity_dict[activity_name] = Dict()
        return activity_dict.get(activity_name)


class ParameterUpdatesAux:

    activities_key = "activities"
    default_key = "default"

    @staticmethod
    def _keep_values(parameter_map, keep_values_set):
        cull_keys = {key for key in parameter_map if key not in keep_values_set}
        for key in cull_keys:
            del parameter_map[key]

    @staticmethod
    def cull_values(parameter_updates, keep_values_set):
        ParameterUpdatesAux._keep_values(
            parameter_updates.update_map__.get(ParameterUpdatesAux.default_key), keep_values_set
        )

        for parameter_map in parameter_updates.update_map__.get(ParameterUpdatesAux.activities_key).values():
            ParameterUpdatesAux._keep_values(parameter_map, keep_values_set)

    @staticmethod
    def get_activity_parameters(parameter_updates, activity_name):
        retval = Dict(parameter_updates.update_map__.get(ParameterUpdatesAux.default_key))
        retval.update(parameter_updates.update_map__.get(ParameterUpdatesAux.activities_key).get(activity_name, Dict()))
        return retval

    @staticmethod
    def assign_update_map(parameter_updates, new_update_map):
        parameter_updates.__dict__["update_map__"] = new_update_map