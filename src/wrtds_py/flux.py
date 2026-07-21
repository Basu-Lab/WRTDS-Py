import numpy as np

_CMS_TO_KG_PER_DAY = 86.4
_CFS_TO_KG_PER_DAY = 2.44657555

def calc_flux(conc: np.ndarray, q: np.ndarray, flow_unit: str = 'cms') -> np.ndarray:        
    match flow_unit:
        case 'cms':
            factor = _CMS_TO_KG_PER_DAY
        case 'cfs':
            factor = _CFS_TO_KG_PER_DAY
        case _:
            raise ValueError("flow_unit must be either 'cms' or 'cfs'.")

    return conc * q * factor