# File to run the IESA-Sim model
import json
from settings_read import settings_read
from main import main

# Specify version number
version_number = "1.09"

# Welcome message
print(f"Welcome to IESA-Sim v{version_number}")


# Read the settings
settings = settings_read(version_number)
print("Retrieved Settings:")
print(json.dumps(settings, indent=4))


# Run IESA-Sim
main(settings)


# Goodbye message
print("IESA-Sim has finalized the scenario simulation.")
