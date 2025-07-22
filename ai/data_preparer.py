import uproot
import numpy as np
import pandas as pd
import vector

print("Starting AI Data Preparation...")

# --- 1. Define Constants and Paths ---
input_file_path = "data/pp_13TeV_5M_events.root"
output_file_path = "data/ai_training_data.csv"
batch_size = 100_000

# Define the mass windows for signal and background labeling
SIGNAL_LOW = 80.0
SIGNAL_HIGH = 100.0
SIDEBAND_LOW_1 = 60.0
SIDEBAND_HIGH_1 = 75.0
SIDEBAND_LOW_2 = 105.0
SIDEBAND_HIGH_2 = 120.0

# This list will store all the processed events (as dictionaries)
labeled_events = []

# --- 2. The Batch Processing Loop ---
try:
    # uproot.iterate() reads the file in chunks (batches)
    for batch in uproot.iterate(
        f"{input_file_path}:events",
        ["particle_id", "particle_px", "particle_py", "particle_pz", "particle_E"],
        step_size=batch_size,
        library="np"
    ):
        print(f"  ... processing a batch of {len(batch['particle_id'])} events")
        
        num_events_in_batch = len(batch["particle_id"])

        for i_event in range(num_events_in_batch):
            pids = batch["particle_id"][i_event]
            pxs = batch["particle_px"][i_event]
            pys = batch["particle_py"][i_event]
            pzs = batch["particle_pz"][i_event]
            energies = batch["particle_E"][i_event]

            muon_indices = np.where(pids == 13)[0]
            antimuon_indices = np.where(pids == -13)[0]

            if len(muon_indices) > 0 and len(antimuon_indices) > 0:
                for i_muon in muon_indices:
                    for i_antimuon in antimuon_indices:
                        muon_vec = vector.obj(px=pxs[i_muon], py=pys[i_muon], pz=pzs[i_muon], E=energies[i_muon])
                        antimuon_vec = vector.obj(px=pxs[i_antimuon], py=pys[i_antimuon], pz=pzs[i_antimuon], E=energies[i_antimuon])
                        
                        mass = (muon_vec + antimuon_vec).mass
                        
                        label = None
                        # Check if the event is in the SIGNAL window
                        if SIGNAL_LOW <= mass <= SIGNAL_HIGH:
                            label = 1
                        # Check if the event is in one of the BACKGROUND sidebands
                        elif (SIDEBAND_LOW_1 <= mass <= SIDEBAND_HIGH_1) or \
                             (SIDEBAND_LOW_2 <= mass <= SIDEBAND_HIGH_2):
                            label = 0
                        
                        # If a label was assigned, structure the data and save it
                        if label is not None:
                            event_features = {
                                'muon_px': muon_vec.px,
                                'muon_py': muon_vec.py,
                                'muon_pz': muon_vec.pz,
                                'muon_E': muon_vec.E,
                                'antimuon_px': antimuon_vec.px,
                                'antimuon_py': antimuon_vec.py,
                                'antimuon_pz': antimuon_vec.pz,
                                'antimuon_E': antimuon_vec.E,
                                'label': label
                            }
                            labeled_events.append(event_features)
                            
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_file_path}")
    exit()

print("\nAll batches processed.")
print(f"Found {len(labeled_events)} total labeled events for AI training.")

# --- 3. Final Conversion and Save to CSV ---
if not labeled_events:
    print("⚠️ No labeled events were found. Cannot create training data file.")
else:
    print("Converting to Pandas DataFrame and saving to CSV...")
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(labeled_events)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)
    
    print(f"✅ Success! AI training data saved to {output_file_path}")
    # You can print the first few rows to check it
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # You can also print the balance of signal vs background
    print("\nDataset balance:")
    print(df['label'].value_counts())