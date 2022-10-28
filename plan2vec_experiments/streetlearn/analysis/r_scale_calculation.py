scale_dict = {
    "Tiny": [524.10825964, 697.14084164],
    "Small": [251.83783118, 286.49520676],
    "Medium": [154.28508823, 210.69826202],
    "Large": [100.0057638, 125.21619651],
    "xl": [50.01646301, 62.51763628]
}

import numpy as np

identity = np.ones(2)
for k, v in scale_dict.items():
    _ = identity / v * scale_dict['Tiny']
    print(f"{k}:", _, 0.2 / _.mean())
