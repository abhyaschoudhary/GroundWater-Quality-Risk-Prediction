"""
generate_sample_data.py
------------------------
Agar Kaggle dataset download karne mein time lag raha hai, ya us dataset
ka format alag hai, to yeh script realistic dummy groundwater data
generate kar dega taaki tum pura pipeline (project.py) turant test kar sako.

IMPORTANT: Yeh sirf TESTING ke liye hai. Final submission ke liye
asli Kaggle/CGWB dataset use karna behtar hoga.

Run: python generate_sample_data.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)

states_districts = {
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut"],
    "West Bengal": ["Kolkata", "Howrah", "Malda", "Murshidabad", "Nadia"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Bikaner", "Udaipur", "Ajmer"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Punjab": ["Amritsar", "Ludhiana", "Patiala", "Jalandhar", "Bathinda"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
}

rows = []
for state, districts in states_districts.items():
    for district in districts:
        for year in range(2015, 2024):
            n_samples = np.random.randint(2, 5)
            for _ in range(n_samples):
                fluoride_bias = 1.5 if state == "Rajasthan" else 0.5
                arsenic_bias = 0.03 if state in ["West Bengal", "Bihar"] else 0.005

                ph = np.round(np.random.normal(7.2, 0.6), 2)
                tds = np.round(np.random.normal(800, 400), 1)
                fluoride = np.round(np.abs(np.random.normal(fluoride_bias, 0.8)), 2)
                nitrate = np.round(np.abs(np.random.normal(30, 25)), 2)
                arsenic = np.round(np.abs(np.random.normal(arsenic_bias, 0.02)), 4)
                iron = np.round(np.abs(np.random.normal(0.4, 0.5)), 2)

                rows.append({
                    "state": state,
                    "district": district,
                    "year": year,
                    "ph": max(5.5, min(9.5, ph)),
                    "tds": max(50, tds),
                    "fluoride": fluoride,
                    "nitrate": nitrate,
                    "arsenic": arsenic,
                    "iron": iron,
                })

df = pd.DataFrame(rows)
output_path = "data/raw/groundwater_sample.csv"
df.to_csv(output_path, index=False)

print(f"Sample dataset ban gaya: {output_path}")
print(f"Total rows: {len(df)}")
print(df.head())
