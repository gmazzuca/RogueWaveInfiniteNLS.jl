from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


BASE_DIR = Path(__file__).parent


def _load_df(csv_path: Path) -> pd.DataFrame:
	"""Load a CSV and return a DataFrame with columns ['X', 'PsiAbs'].
	Try with header first, then fall back to header=None if necessary.
	If the X column contains entries in the form 'int/int' we replace the X column
	with a uniform grid from -4 to 4 spaced by 0.01 (or with a linspace matching
	the number of rows if lengths don't match).
	"""
	# Try reading with pandas default (header inference)
	try:
		df = pd.read_csv(csv_path)
		if set(["X", "PsiAbs"]).issubset(df.columns):
			df = df[["X", "PsiAbs"]].copy()
			 # If X column contains fraction strings like '1/100', replace with uniform grid
			x_as_str = df["X"].astype(str)
			if x_as_str.str.contains('/').any():
				# preferred uniform grid
				grid = np.arange(-4.0, 4.0 + 0.01, 0.01)
				if len(df) == len(grid):
					df["X"] = grid
				else:
					# fallback: match number of rows
					df["X"] = np.linspace(-4.0, 4.0, len(df))
			else:
				# convert to numeric where possible
				df["X"] = pd.to_numeric(df["X"], errors='coerce')
			return df
	except Exception:
		# fall through to fallback
		pass

	# Fallback: read without header and assume first two columns are X and PsiAbs
	df = pd.read_csv(csv_path, header=None)
	if df.shape[1] >= 2:
		df = df.iloc[:, :2].copy()
		df.columns = ["X", "PsiAbs"]
		 # If X column contains fraction strings like '1/100', replace with uniform grid
		x_as_str = df["X"].astype(str)
		if x_as_str.str.contains('/').any():
			grid = np.arange(-4.0, 4.0 + 0.01, 0.01)
			if len(df) == len(grid):
				df["X"] = grid
			else:
				df["X"] = np.linspace(-4.0, 4.0, len(df))
		else:
			# convert to numeric where possible
			df["X"] = pd.to_numeric(df["X"], errors='coerce')
		return df

	raise ValueError(f"Unable to read CSV or find X/PsiAbs columns in: {csv_path}")


def _plot_group(glob_pattern: str, rhp_filename: str, outname: str, title: str, diff_outname: str, diff_title: str):
	"""Plot all files matching glob_pattern in BASE_DIR.
	- The RHP file (rhp_filename) is plotted first as a dashed black line and labeled 'PIII' or 'PV'.
	- Soliton files of the form '{N}SolitonsPIII.csv' or '{N}SolitonsPV.csv' are labeled '{N}-soliton'.
	- Additionally creates a log10 difference plot between the RHP file and each soliton file, saved to diff_outname.
	"""
	files = sorted([p.name for p in BASE_DIR.glob(glob_pattern)])
	if not files:
		print(f"No files found for pattern: {glob_pattern}")
		return

	# Put the RHP file first if present
	if rhp_filename in files:
		files.remove(rhp_filename)
		files.insert(0, rhp_filename)

	# Load all data into dict
	data = {}
	for fname in files:
		csv_path = BASE_DIR / fname
		try:
			df = _load_df(csv_path)
		except Exception as e:
			print(f"Skipping {fname}: {e}")
			continue
		# drop NaNs and sort by X
		df = df.dropna(subset=["X", "PsiAbs"]).sort_values("X")
		data[fname] = df.reset_index(drop=True)

	if rhp_filename not in data:
		print(f"RHP file {rhp_filename} not found or failed to load; skipping plotting for pattern {glob_pattern}")
		return

	# Helper to build label from filename
	def _label_from_name(fname: str, rhp_name: str):
		# soliton files start with number
		m = re.match(r"^(\d+)", fname)
		if m:
			return f"{m.group(1)}-soliton"
		# RHP files labeled by PIII or PV
		if fname == rhp_name:
			if "PIII" in fname:
				return "PIII"
			if "PV" in fname:
				return "PV"
		# fallback to filename
		return fname

	# Main overlay plot
	plt.figure(figsize=(10, 6))
	for idx, fname in enumerate(files):
		if fname not in data:
			continue
		df = data[fname]
		label = _label_from_name(fname, rhp_filename)
		if idx == 0:
			plt.plot(df["X"], df["PsiAbs"], label=label, color="black", linestyle='-.', linewidth=2.5, alpha=0.9)
		else:
			plt.plot(df["X"], df["PsiAbs"], label=label, linewidth=2, alpha=0.4)

	plt.xlabel("X", fontsize=20)
	plt.ylabel(r"$|\psi|$", fontsize=20, labelpad=20,rotation=np.pi/2)
	plt.title(title, fontsize=25)
	plt.xlim(-5, 5)
	plt.legend(fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tight_layout()
	plt.savefig(BASE_DIR / outname)
	plt.show()
	plt.close()

	# Difference log plot: RHP vs each soliton
	rhp_df = data[rhp_filename]
	rhp_x = rhp_df["X"].values
	rhp_y = rhp_df["PsiAbs"].values
	# ensure increasing x for interpolation
	rhp_order = np.argsort(rhp_x)
	rhp_x_sorted = rhp_x[rhp_order]
	rhp_y_sorted = rhp_y[rhp_order]

	plt.figure(figsize=(10, 6))
	for fname in files[1:]:
		if fname not in data:
			continue
		sol_df = data[fname]
		sol_x = sol_df["X"].values
		sol_y = sol_df["PsiAbs"].values
		# interpolate RHP onto soliton X grid
		try:
			rhp_interp = np.interp(sol_x, rhp_x_sorted, rhp_y_sorted)
		except Exception as e:
			print(f"Interpolation failed for {fname}: {e}")
			continue
		delta = np.abs(rhp_interp - sol_y)
		# avoid log10(0)
		delta_safe = np.maximum(delta, np.finfo(float).tiny)
		logdelta = np.log10(delta_safe)
		label = _label_from_name(fname, rhp_filename)
		plt.plot(sol_x, logdelta, label=label, linewidth=1.5, alpha=0.8)

	plt.xlabel("X", fontsize=14)
	plt.ylabel(r"log10 |RHP - soliton|", fontsize=14, labelpad=18)
	plt.title(diff_title, fontsize=16)
	plt.xlim(-5, 5)
	plt.legend(fontsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tight_layout()
	plt.savefig(BASE_DIR / diff_outname)
	plt.show()
	plt.close()


# Plot all P-III related CSVs and produce difference plot
_plot_group("*PIII.csv", "RHP_PIII.csv", "RogueWaveSlice_PIII_T0.png", "Rogue Wave P-III at T=0", "RogueWaveSlice_PIII_diff_log.png", "Log difference: RHP vs P-III solitons at T=0")

# Plot all P-V related CSVs and produce difference plot
_plot_group("*PV.csv", "RHP_PV.csv", "RogueWaveSlice_PV_T0.png", "Rogue Wave P-V at T=0", "RogueWaveSlice_PV_diff_log.png", "Log difference: RHP vs P-V solitons at T=0")

