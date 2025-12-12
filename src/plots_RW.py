from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Collect all available soliton CSVs and the baseline output.
BASE_DIR = Path(__file__).parent
DATASETS = [
	("RogueWaveInfiniteNLS_Output.csv", True, r"$|\psi_{III}(X,0)|$"),
	("50solitonPIII.csv", False, "50-soliton PIII"),
	("100solitonPIII.csv", False, "100-soliton PIII"),
	("120solitonPIII.csv", False, "120-soliton PIII"),
	("1000solitonPIII.csv", False, "1000-soliton PIII"),
]


plt.figure(figsize=(10, 6))

for idx, (filename, has_header, label) in enumerate(DATASETS):
	csv_path = BASE_DIR / filename
	if has_header:
		df = pd.read_csv(csv_path)
	else:
		df = pd.read_csv(csv_path, header=None, names=["X", "PsiAbs"])

	if idx == 0:
		plt.plot(df["X"], df["PsiAbs"], label=label, alpha=0.7, linestyle='--', linewidth=2.5)
	else:
		plt.plot(df["X"], df["PsiAbs"], label=label, alpha=0.7, linewidth=1.5)

plt.xlabel("X", fontsize=14)
plt.ylabel(r"$|\psi|$", fontsize=14)
plt.title("Rogue Wave P-III at T=0", fontsize=16)
plt.xlim(-5, 5)
plt.legend(fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(BASE_DIR / "RogueWaveSlice_T0.png")
plt.show()

