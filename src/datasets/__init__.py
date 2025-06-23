from .gaussian_ns import  GaussianNS
from .gaussian_ns1 import  GaussianNS1
from .gaussian_ns2 import  GaussianNS2
from .custom import CustomCsv

import builtins
builtins.Custom = CustomCsv

# Alias for single-file custom CSV loader
Custom = CustomCsv

# lowercase alias so `dataset_type="custom"` works with parse_type
builtins.custom = CustomCsv
custom = CustomCsv

from .customMulty import CustomMultyCsv, CustomMultyLoader
