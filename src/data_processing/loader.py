import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
def normalize_quaternion(q):
    """
    Normalize a single quaternion to unit length.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion encountered")
    return q / norm
def enforce_antipodal_continuity(quaternions):
    """
    Fix antipodal discontinuities in a sequence of quaternions.
    If dot(q_t, q_{t-1}) < 0, flip q_t.
    """
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i-1], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    return quaternions
def load_quaternion_sequence(file_path):
    """
    Load head tracking data from a file and return a cleaned
    sequence of qu
aternions with normalization and antipodal fix.
    """
    print(f"Loading file: {file_path}")
    
    # Load the file
    df = pd.read_csv(file_path)
    df.columns = ["idx", "longitude", "latitude", "timestamp"]
    print("First 5 rows of the data:")
    print(df.head())
    print("Data summary:")
    print(df.describe())
    
    # Convert lon/lat to quaternions (roll=0)
    lon = df["longitude"].values
    lat = df["latitude"].values
    angles = np.vstack([lon, lat, np.zeros_like(lon)]).T
    
    rot = R.from_euler('YXZ', angles, degrees=False)
    quaternions = rot.as_quat()  # [x, y, z, w]
    print("85 quaternions (raw):")
    print(quaternions[:85])
    
    # Normalize each quaternion
    quaternions = np.array([normalize_quaternion(q) for q in quaternions])
    print("85 quaternions after normalization:")
    print(quaternions[:85])
    
    # Enforce antipodal continuity
    quaternions = enforce_antipodal_continuity(quaternions)
    print("85 quaternions after antipodal fix:")
    print(quaternions[:85])
    
    return quaternions
if __name__ == "__main__":
    # Example usage: load the first file in the Scanpaths folder
    data_path = "data/images/Scanpaths"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    if files:
        file_to_load = os.path.join(data_path, files[0])
        quats = load_quaternion_sequence(file_to_load)
	# Create a DataFrame
        quat_df = pd.DataFrame(quats, columns=["x", "y", "z", "w"])
        
        # Save to CSV
        output_csv = "cleaned_quaternions.csv"
        quat_df.to_csv(output_csv, index=False)
        print(f"Cleaned quaternions saved to {output_csv}")
