# import uproot
# import numpy as np
# import matplotlib.pyplot as plt
# import vector

# print("Starting data processing and validation...")

# # --- 1. FILE AND DATA LOADING ---
# # Path to the ROOT file created by our simulation.
# input_file_path = "data/pp_13TeV_5M_events.root"

# # Load the data using uproot.
# # We specify the branches (columns) we need to minimize memory usage.
# try:
#     with uproot.open(input_file_path) as file:
#         tree = file["events"]
#         branches = tree.arrays(["particle_id", "particle_px", "particle_py", "particle_pz", "particle_E"], library="np")
#         print("✅ Data loaded successfully.")
# except FileNotFoundError:
#     print(f"❌ Error: Input file not found at {input_file_path}")
#     print("Please run collision_generator.py first.")
#     exit()

# # --- 2. DATA PROCESSING: FIND Z BOSON CANDIDATES ---
# invariant_masses = []
# print("Searching for Z boson candidates (muon-antimuon pairs)...")

# # Get the total number of events
# num_events = len(branches["particle_id"])

# for i_event in range(num_events):
#     # Get the particle data for the current event
#     pids = branches["particle_id"][i_event]
#     pxs = branches["particle_px"][i_event]
#     pys = branches["particle_py"][i_event]
#     pzs = branches["particle_pz"][i_event]
#     energies = branches["particle_E"][i_event]

#     # Find the indices of muons (id=13) and antimuons (id=-13)
#     muon_indices = np.where(pids == 13)[0]
#     antimuon_indices = np.where(pids == -13)[0]

#     # If we have at least one of each, check all possible pairs
#     if len(muon_indices) > 0 and len(antimuon_indices) > 0:
#         for i_muon in muon_indices:
#             for i_antimuon in antimuon_indices:
#                 # Create 4D momentum vectors for the muon and antimuon
#                 # Using the 'vector' library which simplifies calculations
#                 muon_vec = vector.obj(px=pxs[i_muon], py=pys[i_muon], pz=pzs[i_muon], E=energies[i_muon])
#                 antimuon_vec = vector.obj(px=pxs[i_antimuon], py=pys[i_antimuon], pz=pzs[i_antimuon], E=energies[i_antimuon])

#                 # The total momentum is the sum of the two vectors
#                 z_boson_candidate = muon_vec + antimuon_vec

#                 # The invariant mass is the 'mass' property of the resulting vector
#                 # The library handles the formula: sqrt(E^2 - px^2 - py^2 - pz^2)
#                 mass = z_boson_candidate.mass
#                 invariant_masses.append(mass)
    
#     # Progress indicator
#     if (i_event + 1) % 1000 == 0:
#         print(f"  ... processed {i_event + 1} / {num_events} events")

# print(f"Found {len(invariant_masses)} total muon-antimuon pairs.")

# # --- 3. VISUALIZATION ---
# if not invariant_masses:
#     print("⚠️ No muon-antimuon pairs found. Cannot generate plot.")
#     exit()

# print("Generating invariant mass plot...")
# output_plot_path = "visualization/plots/invariant_mass_z_boson_5M.png"

# # The known mass of the Z boson in GeV for reference
# Z_BOSON_MASS = 91.1876

# plt.figure(figsize=(12, 8))
# # Create the histogram
# # We'll focus the plot range from 50 to 150 GeV to see the Z boson peak clearly
# plt.hist(invariant_masses, bins=30, range=(50, 150), color='royalblue', alpha=0.7)

# # Add a vertical line at the official Z boson mass
# plt.axvline(x=Z_BOSON_MASS, color='red', linestyle='--', linewidth=2, label=f'Known Z Boson Mass ({Z_BOSON_MASS} GeV)')

# # Add titles and labels
# plt.title('Invariant Mass of Muon-Antimuon Pairs', fontsize=18)
# plt.xlabel('Invariant Mass [GeV]', fontsize=14)
# plt.ylabel('Number of Candidate Events', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()

# # Save the plot
# plt.savefig(output_plot_path)
# print(f"✅ Success! Plot saved to {output_plot_path}")
# plt.show() # Optional: display the plot directly if you have a GUI




import uproot
import numpy as np
import matplotlib.pyplot as plt
import vector

print("Starting data processing and validation (Memory-Efficient Mode)...")

# --- 1. FILE AND BATCHING SETUP ---
# Path to the large ROOT file.
input_file_path = "data/pp_13TeV_5M_events.root"

# We will process the file in chunks of 100,000 events to keep memory usage low.
batch_size = 100_000
invariant_masses = []

print(f"Preparing to process {input_file_path} in batches of {batch_size} events.")

# --- 2. BATCH PROCESSING LOOP ---
try:
    # uproot.iterate() reads the file in chunks (batches).
    # The 'step_size' argument controls how many events are in each chunk.
    for batch in uproot.iterate(
        f"{input_file_path}:events",
        ["particle_id", "particle_px", "particle_py", "particle_pz", "particle_E"],
        step_size=batch_size,
        library="np"
    ):
        print(f"  ... processing a batch of {len(batch['particle_id'])} events")
        
        # Get the total number of events in the current batch
        num_events_in_batch = len(batch["particle_id"])

        for i_event in range(num_events_in_batch):
            # Get the particle data for the current event from the batch
            pids = batch["particle_id"][i_event]
            pxs = batch["particle_px"][i_event]
            pys = batch["particle_py"][i_event]
            pzs = batch["particle_pz"][i_event]
            energies = batch["particle_E"][i_event]

            # Find the indices of muons (id=13) and antimuons (id=-13)
            muon_indices = np.where(pids == 13)[0]
            antimuon_indices = np.where(pids == -13)[0]

            # If we have at least one of each, check all possible pairs
            if len(muon_indices) > 0 and len(antimuon_indices) > 0:
                for i_muon in muon_indices:
                    for i_antimuon in antimuon_indices:
                        # Create 4D momentum vectors
                        muon_vec = vector.obj(px=pxs[i_muon], py=pys[i_muon], pz=pzs[i_muon], E=energies[i_muon])
                        antimuon_vec = vector.obj(px=pxs[i_antimuon], py=pys[i_antimuon], pz=pzs[i_antimuon], E=energies[i_antimuon])
                        
                        # Calculate the invariant mass of the pair
                        z_boson_candidate = muon_vec + antimuon_vec
                        mass = z_boson_candidate.mass
                        invariant_masses.append(mass)

except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_file_path}")
    print("Please ensure the 5M event file was generated correctly.")
    exit()

print("\nAll batches processed.")
print(f"Found {len(invariant_masses)} total muon-antimuon pairs.")


# --- 3. VISUALIZATION ---
if not invariant_masses:
    print("⚠️ No muon-antimuon pairs found. Cannot generate plot.")
    exit()

print("Generating final invariant mass plot...")
output_plot_path = "visualization/plots/invariant_mass_z_boson_5M.png"

Z_BOSON_MASS = 91.1876

plt.figure(figsize=(12, 8))
# Use 30 wider bins to better visualize the peak
plt.hist(invariant_masses, bins=30, range=(50, 150), color='royalblue', alpha=0.7)
plt.axvline(x=Z_BOSON_MASS, color='red', linestyle='--', linewidth=2, label=f'Known Z Boson Mass ({Z_BOSON_MASS} GeV)')
plt.title('Invariant Mass of Muon-Antimuon Pairs (5 Million Events)', fontsize=18)
plt.xlabel('Invariant Mass [GeV]', fontsize=14)
plt.ylabel('Number of Candidate Events', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig(output_plot_path)

print(f"✅ Success! Plot saved to {output_plot_path}")
plt.show()