import sys
print(">>> PYTHON RUNNING FROM:")
print(sys.executable)

import pkgutil
print("category_encoders installed?", "category_encoders" in [p.name for p in pkgutil.iter_modules()])