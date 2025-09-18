from genesis.repr_base import RBC

class Dimensional(RBC):
    def __init__(self, dimensionless_vars):
        self._dimensionless_vars = set(dimensionless_vars)

    def to_units(self, units):
        for attr_name, attr_value in vars(self).items():
            if attr_name in self.dimensionless_vars: continue
            if hasattr(attr_value, 'to_units'): attr_value.to_units(units)
            elif hasattr(attr_value, 'to'): attr_value.to(units)
            else: raise AttributeError(f"Object for attribute '{attr_name}' has no method for converting units.")

    @property
    def dimensionless_vars(self):
        return self._dimensionless_vars