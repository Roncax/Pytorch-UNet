import json

import paths

colormap = json.load(open(paths.json_file))['colormap']
print(colormap["0"][1])