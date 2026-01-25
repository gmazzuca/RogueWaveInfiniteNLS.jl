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
	- Soliton files of the form '{N}SolitonsPIII.csv' or '{N}SolitonsPV.csv' are labeled '{N}-soliton' and
	  are plotted in numeric order (e.g. 50, 100, 200).
	- Additionally creates a log10 difference plot between the RHP file and each soliton file, saved to diff_outname,
	  and a separate L2-norm vs N plot saved alongside.
	"""
	files_all = sorted([p.name for p in BASE_DIR.glob(glob_pattern)])
	if not files_all:
		print(f"No files found for pattern: {glob_pattern}")
		return

	# Separate RHP and soliton/other files
	soliton_re = re.compile(r"^(\d+)")
	rhp = rhp_filename
	others = [f for f in files_all if f != rhp]
	# identify soliton files (those starting with a number)
	solitons = [f for f in others if soliton_re.match(f)]
	others_non_soliton = [f for f in others if f not in solitons]
	# sort solitons by numeric prefix (ensures 50,100,200 order)
	solitons_sorted = sorted(solitons, key=lambda f: int(soliton_re.match(f).group(1)))

	# Build final ordered list: RHP first (if present), then sorted solitons, then any other files
	files = []
	if rhp in files_all:
		files.append(rhp)
	files.extend(solitons_sorted)
	files.extend(others_non_soliton)

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

	if rhp not in data:
		print(f"RHP file {rhp} not found or failed to load; skipping plotting for pattern {glob_pattern}")
		return

	# Helper to build label from filename
	def _label_from_name(fname: str, rhp_name: str):
		m = soliton_re.match(fname)
		if m:
			return f"{m.group(1)}-soliton"
		if fname == rhp_name:
			if "PIII" in fname:
				return "PIII"
			if "PV" in fname:
				return "PV"
		return fname

	# Determine dataset label for use in y-labels (PIII or PV)
	dataset_label = "PIII" if "PIII" in rhp else ("PV" if "PV" in rhp else "")

	# Main overlay plot
	plt.figure(figsize=(10, 6))
	for idx, fname in enumerate(files):
		if fname not in data:
			continue
		df = data[fname]
		label = _label_from_name(fname, rhp)
		if idx == 0:
			plt.plot(df["X"], df["PsiAbs"], label=label, color="black", linestyle='-.', linewidth=2.5, alpha=0.9)
		else:
			plt.plot(df["X"], df["PsiAbs"], label=label, linewidth=2, alpha=0.4)

	plt.xlabel("X", fontsize=20)
	plt.ylabel(r"$|\psi|$", fontsize=20, labelpad=20)
	plt.title(title, fontsize=25)
	plt.xlim(-5, 5)
	plt.legend(fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tight_layout()
	plt.savefig(BASE_DIR / outname)
	plt.show()
	plt.close()

	# Difference log plot: RHP vs each soliton
	rhp_df = data[rhp]
	rhp_x = rhp_df["X"].values
	rhp_y = rhp_df["PsiAbs"].values
	# ensure increasing x for interpolation
	rhp_order = np.argsort(rhp_x)
	rhp_x_sorted = rhp_x[rhp_order]
	rhp_y_sorted = rhp_y[rhp_order]

	plt.figure(figsize=(10, 6))
	# prepare L2 collections
	Ns = []
	L2_vals = []
	for fname in solitons_sorted:
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
		# compute L2 norm: sqrt(integral delta^2 dx)
		l2_sq = np.trapezoid(delta**2, sol_x)
		l2 = np.sqrt(l2_sq)
		# collect N and L2
		n_match = soliton_re.match(fname)
		N = int(n_match.group(1)) if n_match else None
		if N is not None:
			Ns.append(N)
			L2_vals.append(l2)
		# avoid log10(0)
		delta_safe = np.maximum(delta, np.finfo(float).tiny)
		logdelta = np.log10(delta_safe)
		label = _label_from_name(fname, rhp)
		plt.plot(sol_x, logdelta, label=label, linewidth=1.5, alpha=0.8)

	ylabel = rf"$\log_{{10}} |\Psi_{{{dataset_label}}} - \Psi_{{N,{dataset_label}}}|$"
	plt.xlabel("X", fontsize=14)
	plt.ylabel(ylabel, fontsize=14, labelpad=18)
	plt.title(diff_title, fontsize=16)
	plt.xlim(-5, 5)
	plt.legend(fontsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tight_layout()
	plt.savefig(BASE_DIR / diff_outname)
	plt.show()
	plt.close()

	# Plot L2 norm vs N (number of solitons)
	if Ns:
		# sort by N
		order = np.argsort(Ns)
		Ns_sorted = np.array(Ns)[order]
		L2_sorted = np.array(L2_vals)[order]
		plt.figure(figsize=(8, 5))
		plt.semilogy(Ns_sorted, L2_sorted, marker='o', linestyle='-', linewidth=2)
		plt.xlabel('Number of solitons (N)', fontsize=14)
		# reuse the same ylabel as the log-difference plot (dataset_label is in scope)
		# the variable `ylabel` was built earlier for the log plot and includes the dataset name
		try:
			plt.ylabel(ylabel, fontsize=14, labelpad=12)
		except NameError:
			# fallback if ylabel not defined for some reason
			plt.ylabel(r'$\|\Psi_{\mathrm{RHP}} - \Psi_{N}\|_{2}$', fontsize=14)
		plt.title(rf'$L_2$ difference ({dataset_label})', fontsize=16)
		# set x-ticks at the Ns values with labels
		plt.xticks(Ns_sorted, [str(int(n)) for n in Ns_sorted], fontsize=12)
		# set y-ticks at powers of ten covering the L2 range
		# handle zero or extremely small values
		pos_vals = L2_sorted[L2_sorted > 0]
		if pos_vals.size > 0:
			kmin = int(np.floor(np.log10(pos_vals.min())))
			kmax = int(np.ceil(np.log10(L2_sorted.max())))
			yticks = 10.0 ** np.arange(kmin, kmax + 1)
			ytick_labels = [f'$10^{{{k}}}$' for k in range(kmin, kmax + 1)]
			plt.yticks(yticks, ytick_labels, fontsize=12)
		# minor grid and styling
		plt.grid(True, which='both', ls='--', alpha=0.4)
		plt.tight_layout()
		# derive L2 output name from diff_outname
		l2_outname = Path(diff_outname).stem + '_L2.png'
		plt.savefig(BASE_DIR / l2_outname)
		plt.show()
		plt.close()


def _plot_averaged_vs_rhp():
	"""Plot RHP_PIII.csv (dashed) against averaged soliton files (solid lines).
	Also creates a log L2 difference plot.
	"""
	# Load RHP data from data/ directory
	rhp_path = BASE_DIR.parent / "data" / "RHP_PIII.csv"
	try:
		rhp_df = _load_df(rhp_path)
	except Exception as e:
		print(f"Failed to load RHP file: {e}")
		return
	
	# Files to plot (in src/ directory)
	averaged_files = [
		"50solitonPIII_AVERAGED.csv",
		"100solitonPIII_AVERAGED.csv",
		"200solitonPIII_AVERAGED.csv"
	]
	
	# Load averaged data - use RHP's X column for all
	data = {}
	rhp_x = rhp_df["X"].values
	rhp_y = rhp_df["PsiAbs"].values
	
	for fname in averaged_files:
		csv_path = BASE_DIR / fname
		try:
			df = pd.read_csv(csv_path, header=None)
			# Skip first column (string), use second column for PsiAbs
			if df.shape[1] >= 2:
				psi_vals = pd.to_numeric(df.iloc[:, 1], errors='coerce')
				# Create DataFrame with RHP's X values
				new_df = pd.DataFrame({
					"X": rhp_x[:len(psi_vals)],
					"PsiAbs": psi_vals.values
				})
				data[fname] = new_df.dropna()
			else:
				print(f"Skipping {fname}: insufficient columns")
		except Exception as e:
			print(f"Skipping {fname}: {e}")
	
	if not data:
		print("No averaged files loaded successfully")
		return
	
	# Main overlay plot
	plt.figure(figsize=(10, 6))
	
	# Plot RHP with dashed line
	plt.plot(rhp_x, rhp_y, label="RHP PIII", color="black", linestyle="--", linewidth=2.5, alpha=0.9)
	
	# Plot averaged files with solid lines
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
	labels = ['50-soliton averaged', '100-soliton averaged', '200-soliton averaged']
	
	for idx, (fname, label) in enumerate(zip(averaged_files, labels)):
		if fname in data:
			df = data[fname]
			plt.plot(df["X"], df["PsiAbs"], label=label, color=colors[idx], linewidth=2, alpha=0.7)
	
	plt.xlabel("X", fontsize=20)
	plt.ylabel(r"$|\Psi|$", fontsize=20, labelpad=20, rotation=0)
	plt.title(r"$\Psi_{III}(X,0)$ vs $\Psi_{N,III}(X,0)$", fontsize=25)
	plt.xlim(-5, 5)
	plt.legend(fontsize=16)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tight_layout()
	plt.savefig(BASE_DIR / "RHP_PIII_vs_Averaged.png")
	plt.show()
	plt.close()
	
	# Log L2 difference plot
	plt.figure(figsize=(10, 6))
	
	# Prepare for L2 calculation
	Ns = []
	L2_vals = []
	
	rhp_order = np.argsort(rhp_x)
	rhp_x_sorted = rhp_x[rhp_order]
	rhp_y_sorted = rhp_y[rhp_order]
	
	for idx, (fname, label) in enumerate(zip(averaged_files, labels)):
		if fname not in data:
			continue
		
		df = data[fname]
		sol_x = df["X"].values
		sol_y = df["PsiAbs"].values
		
		# Interpolate RHP onto soliton X grid
		try:
			rhp_interp = np.interp(sol_x, rhp_x_sorted, rhp_y_sorted)
		except Exception as e:
			print(f"Interpolation failed for {fname}: {e}")
			continue
		
		delta = np.abs(rhp_interp - sol_y)
		
		# Compute L2 norm
		l2_sq = np.trapezoid(delta**2, sol_x)
		l2 = np.sqrt(l2_sq)
		
		# Extract N from filename
		N = int(fname.split('soliton')[0])
		Ns.append(N)
		L2_vals.append(l2)
		
		# Plot log difference
		delta_safe = np.maximum(delta, np.finfo(float).tiny)
		logdelta = np.log10(delta_safe)
		plt.plot(sol_x, logdelta, label=label, color=colors[idx], linewidth=2, alpha=0.8)
	
	plt.xlabel("X", fontsize=20)
	plt.ylabel(r"$\log_{10} |\Psi_{III}(X,0) - \Psi_{N,III}(X,0)|$", fontsize=20, labelpad=20)
	plt.title("Log L2 Difference: RHP PIII vs Averaged Solitons", fontsize=25)
	plt.xlim(-5, 5)
	plt.legend(fontsize=16)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tight_layout()
	plt.savefig(BASE_DIR / "RHP_PIII_vs_Averaged_diff_log.png")
	plt.show()
	plt.close()
	
	# Plot L2 norm vs N
	if Ns:
		order = np.argsort(Ns)
		Ns_sorted = np.array(Ns)[order]
		L2_sorted = np.array(L2_vals)[order]
		
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.semilogy(Ns_sorted, L2_sorted, marker='o', linestyle='-', linewidth=2, markersize=8)
		ax.set_xlabel(r'Number of solitons ($N$)', fontsize=15)
		ax.set_ylabel(r'$\left|\left|\Psi_{III} - \Psi_{N,III}\right|\right|_2$', fontsize=15, rotation=0)
		ax.yaxis.set_label_coords(-0.15, 0.4)
		ax.set_title(r'$L_2$ Difference: $\Psi_{III}(X,0)$ vs $\Psi_{N,III}(X,0)$', fontsize=25)
		ax.set_xticks(Ns_sorted)
		ax.set_xticklabels([str(int(n)) for n in Ns_sorted], fontsize=15)
		
		# Add more y-ticks at powers of 10
		pos_vals = L2_sorted[L2_sorted > 0]
		if pos_vals.size > 0:
			kmin = int(np.floor(np.log10(pos_vals.min())))
			kmax = int(np.ceil(np.log10(L2_sorted.max())))
			yticks = 10.0 ** np.arange(kmin, kmax + 1)
			ytick_labels = [f'$10^{{{k}}}$' for k in range(kmin, kmax + 1)]
			ax.set_yticks(yticks)
			ax.set_yticklabels(ytick_labels, fontsize=15)
		
		ax.grid(True, which='both', ls='--', alpha=0.4)
		plt.tight_layout()
		plt.savefig(BASE_DIR / "RHP_PIII_vs_Averaged_L2.png")
		plt.show()
		plt.close()


# # Plot all P-III related CSVs and produce difference plot
# _plot_group("*PIII.csv", "RHP_PIII.csv", "RogueWaveSlice_PIII_T0.png", "Rogue Wave P-III at T=0", "RogueWaveSlice_PIII_diff_log.png", "Log difference: RHP vs P-III solitons at T=0")

# # Plot all P-V related CSVs and produce difference plot
# _plot_group("*PV.csv", "RHP_PV.csv", "RogueWaveSlice_PV_T0.png", "Rogue Wave P-V at T=0", "RogueWaveSlice_PV_diff_log.png", "Log difference: RHP vs P-V solitons at T=0")

# Plot RHP_PIII vs averaged soliton files
_plot_averaged_vs_rhp()

