# File to run the IESA-Sim model
import os
import json
from settings_read import settings_read
from main import main
import matplotlib.pyplot as plt

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

# Specify version number
version_number = "1.10"

# Welcome message
print(f"Welcome to IESA-Sim v{version_number}")

# Read the settings
settings = settings_read(version_number)
print("Retrieved Settings:")
print(json.dumps(settings, indent=4))

# Run IESA-Sim
main(settings)

# Keep plots open even if the model is done running
plt.show()

# Goodbye message
print("IESA-Sim has finalized the scenario simulation.")

