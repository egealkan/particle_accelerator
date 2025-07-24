import sys
# The ROOT library, which we installed with conda. It's the standard in particle physics for data handling.
import ROOT
# The Pythia8 library, also installed with conda.
from pythia8 import Pythia

# --- 1. SETUP ---

# The path to our configuration file.
pythia_config_file = "physics/configs/pp_13TeV_Z_to_muons.cmnd"

# The path where we will save our output data.
output_file_path = "data/signal_Z_to_muons_200k.root"

# The number of collision events we want to generate.
# We start with 10,000 for a quick test run. We can increase this to millions later.
num_events = 200000

print("Initializing PYTHIA...")
# Initialize Pythia with our configuration file.
pythia = Pythia()
pythia.readFile(pythia_config_file)
pythia.init()
print("PYTHIA initialization complete.")


# --- 2. PREPARE THE OUTPUT FILE ---
# We use ROOT's TFile and TTree to store the data efficiently.
# A TTree is like a spreadsheet, where each row is an "event" and columns are particle properties.
print(f"Preparing output file at: {output_file_path}")
output_file = ROOT.TFile(output_file_path, "RECREATE")
tree = ROOT.TTree("events", "Particle Tree")

# Define the "columns" of our spreadsheet. We'll store basic properties for each particle.
# We use arrays because each event will have a different number of particles.
MAX_PARTICLES = 2000 # Set a max limit for array sizes
particle_id = ROOT.std.vector('int')()
particle_status = ROOT.std.vector('int')()
particle_px = ROOT.std.vector('float')()
particle_py = ROOT.std.vector('float')()
particle_pz = ROOT.std.vector('float')()
particle_E = ROOT.std.vector('float')()

# Link our variables to the TTree branches (columns).
tree.Branch("particle_id", particle_id)
tree.Branch("particle_status", particle_status)
tree.Branch("particle_px", particle_px)
tree.Branch("particle_py", particle_py)
tree.Branch("particle_pz", particle_pz)
tree.Branch("particle_E", particle_E)


# --- 3. THE EVENT GENERATION LOOP ---
print(f"Starting event generation for {num_events} events...")

for i_event in range(num_events):
    # Tell Pythia to generate the next collision event.
    if not pythia.next():
        continue

    # Clear the vectors for the new event's data.
    particle_id.clear()
    particle_status.clear()
    particle_px.clear()
    particle_py.clear()
    particle_pz.clear()
    particle_E.clear()

    # Loop through all the particles Pythia generated in this single collision.
    for particle in pythia.event:
        # We only save the "final" particles - the ones that would actually hit a detector.
        if particle.isFinal():
            particle_id.push_back(particle.id())
            particle_status.push_back(particle.statusAbs())
            particle_px.push_back(particle.px())
            particle_py.push_back(particle.py())
            particle_pz.push_back(particle.pz())
            particle_E.push_back(particle.e())

    # Fill the TTree with the data from this event. This is like adding a row to our spreadsheet.
    tree.Fill()

    # Progress indicator
    if (i_event + 1) % 1000 == 0:
        print(f"  ... generated {i_event + 1} / {num_events} events")


# --- 4. FINALIZE AND CLEAN UP ---
print("Event generation complete.")
print("Writing data to ROOT file...")

# Write the TTree to the file and close it.
output_file.Write()
output_file.Close()

# Optional: Print a summary of the simulation.
pythia.stat()

print(f"✅ Success! Data saved to {output_file_path}")