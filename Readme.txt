Contains the computer code to simulate two-trophic phytoplankton communities
based on empirically measured traits and mechanistic species interactions

Author: J.w. Spaak, j.w.spaak@gmail.com
Corrsponding publication:
"Mechanistic models of trophic interactions: opportunities for species richness and challenges for modern coexistence"

##################################################################
# General work flow:
The code is build to simulate a community assembly in a two-trophic plankton community model.
The community assembly is repeated for many different parameter settings, these simulations are performed
via the "sim_*.py" files. These generate "*.npz" files with the raw output (these "*.npz" files are not part of the repository
due to the large data).
The combine_*.py files aggreate the raw-output from the simulations and combine them into *.csv files.
These *.csv files are contained in the repository and are sufficient to reproduce all the plots.
The plots shown in the manuscript can be reproduced with the corresponding plot_*.py files, which generally
create the Figure with the corresponding name.
A complete list of each script and what it creates is listed below.

#################################################################
# Community model scripts:

generate_plankton.py
	Contains functions to create plankton species with traits,
	Most important function is "generate_plankton"
phytoplankton_traits.py
	Loads the empirically measured traits for phytoplankton species
zoop_traits.py
	Loads the empirically measured traits for phytoplankton species
plankton_growth.py
	Contains growth functions of phytoplankton, use either
	"per_cap_plankton_growth" for per capita growth rates
	"plankton_growth" for actual growth rates
	"convert_ode_to_log" for actual growth rates but log converted
assembly_time_fun.py
	Simluates the community assembly.
	Most important function is "assembly_richness"
	For examples on how to use see sim_assembly_mean.py nd similar
NFD_equilibrium.py
	Contains functions to compute niche and fitness differences of two competing
	zooplankton species

######################################################################
# Data files

# empirical data from other literature
growth_rates_brun2017.csv
	Contains the maximum growth rates of zooplankton
	Units of columns:
	"Taxon": -, taxon of species
	"Body mass (mg)": mg, Body mass in miligram C
	"Body mass type": -
	"Growth": mg*h^-1, growth per hour. Brun et al. 2017 reports this as mg*d^-1, however, in Kiorboe 2014 it is listed as mg*h^-1, which believe is the case
	"Specific growth" h^-1, growth rate per mg of species
	Growth (15°): mg*h^-1, Growth scaled to a reference temperature of 15°
	Specific growth (15°): h^-1, Growth rate per mg of species at 15°
	Reference key: Citation of original paper for data
	Original data: https://doi.pangaea.de/10.1594/PANGAEA.862968
clearance_rates_brun2017.csv
	Contains the clearance rates of zooplankton
	Units of columns:
	"Taxon": -, taxon of species
	"Temperature": °C, temperature of experiment
	"Body mass (mg)": mg, Body mass in miligram C
	"Body mass type": -
	"Fmax": \ml*h^-1*ind^-1, maximum clearance rate per individuum
	"Fmax": \ml*h^-1*ind^-1, maximum clearance rate at refernce temperature
	"Specific Fmax": \ml*h^-1*mg^-1, maximum clearance rate per mass of species
	Reference key: Citation of original paper for data
	Original data: https://doi.pangaea.de/10.1594/PANGAEA.862968
Hirst_Kiorboe_2002.csv
	Mortality rates of species. The data of hirst and Kiorboe were not publicly available
	and we have reconstructed this data from the figures in Hirst and Kiroboe 2002.
	"size_Z": log of bodymass in \mug dry weight
	"m_z": d^-1, mortality rate per species in
uye_1989.csv
	Nutrient content of zooplankton species
	The data of Uye 1989 was not publicly available and we have reconstructed this data from the figures in Uye 1989
	Species: Investigated species
	Length: Length of species in mm
	N: nitrogen content in \mug
	C: carbon content in \mug
	DW: dry weight in \mug	
Phytoplankton traits.csv
	Contains most phytoplankton traits.
	Explanation of the columns can be found in "units_phytoplankton_traits.csv"
	Source
	Kyle F. Edwards, Christopher A. Klausmeier, and Elena Litchman. 2015. Nutrient utilization traits of phytoplankton. Ecology 96:2311
units_phytoplankton_traits.csv
	Explanation of the columns of "Phytoplankton traits.csv"
Light_traits.csv
	Contains the light competition traits
	Species: Species name
	Source: Original paper where the data stems from
	mu_l: d^-1, maximum growth rate for model 1
	alpha: quanta* mumol photon^-1 m^2*s, initial slope of growth rate
	AIC_1: Aikake information criterion for model 1
	mu_l_2: d^-1, maximum growth rate for model 1
	alpha_2: quanta* mumol photon^-1 m^2*s, initial slope of growth rate
	I_o: quanta* mumol photon^-1 m^2*s, optimal light intensity
	AIC_2: Aikake information criterion for model 2
	best_model: -, whether model 1 or 2 fits better
	Fiterror: -, How good is the fit
	Notes: Additional information
	Reason: Additional information
	Source: https://aslopubs.onlinelibrary.wiley.com/doi/epdf/10.1002/lno.10282
ehrlich2020.csv
	Contains the defense traits
	Morphotype: Morphtype as identified by the paper
	Morphotype name: Corresponding name
	Defense: -, defensive trait used in paper
	Cell volume: \mu m^3, species bodysize
	r: d^-1, maximum growth rate
	Phosphate: d^-1* \mu mol L^-1Phosphate affinity,	
	Source: Kath, Nadja J.; Gaedke, Ursula; Ehrlich, Elias (2020): Phytoplankton biomass and traits including environmental data 1979-1999 for Lake Constance. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11830464.v1 
augusti_data.csv
	Light absorption coefficient
	Species: Species name
	I: \mu einsteins m^-2 s^-1, irradiance of experiment
	d: \mu m, maximum cell diameter
	Chl: pg, cellular chlorophyll a content
	c_i: pg m^-3, intracellular Chl a content
	a_cell: \mu m^2 cell^-1, cell absorption coefficient
	b_cell: \mu m^2 cell^-1, cell scattering coefficient
	Source: http://www.nrcresearchpress.com/doi/10.1139/f91-091
	


######################################################################
Simulation scripts
sim_assembly_corr.py
	Simulates the community assembly for the altered correlations with increased precision, i.e.
	for the corrlations shown in figure 3 in the paper.
	Generates the files "assembly_corr_20_1000_'trait'_'trait'_'itera'.npz
sim_assembly_long.py
	Simulates the community assembly for longer assembly times, i.e. the data shown in figure S1
	Generates the files "assembly_long_60_1000_'itera'.npz"
sim_assembly_mean.py
	Simulates the community assembly for the altered mean, i.e. the data shown in figure 2
	Generates the files "assembly_mean_20_1000_'trait'_'itera'.npz"
sim_assembly_var.py
	Simulates the community assembly for the altered mean, i.e. the data shown in figure S2
	Generates the files "assembly_var_20_1000_'trait'_'itera'.npz"
sim_assembly_mean.py
	Simulates the community assembly for the altered mean, i.e. the data shown in figure 2
	Generates the files "assembly_mean_20_1000_'trait'_'itera'.npz"
sim_assembly_non_lim.py
	Simuates the community assembly where not all resources are limiting,
	i.e. the data shown in figure S7
	Generates the files "assembly_non_lim_*.npz"
sim_fast_NFD_corr.py
	Computes the niche and fitness differences for the altered correlation communities
	i.e. the data shown in figure 4
	Generates the files "assembly_corr_1000_'trait'_'trait'_'itera'_'itera'.npz"
sim_fast_NFD_mean.py
	Computes the niche and fitness differences for the altered mean values
	i.e. the data shown in figure 4
	Generates the files "sim_NFD_mean_'trait'_'itera'_'itera'.npz"

# The simulations create large datas which are stored, for convenience, in "*.npz" files, which should be opened in python.
These immediate simulation outputs are then combined by the following scripts into the form of csv.
The npz files, because they are large, are not contained in the repository (several GB).
However, the code deposited can recreate these files. If needed, these files can be requested from Jurg Spaak: j.w.spaak@gmail.com
There are more than 100 such *.npz files. In the file names 'trait' implies any of the phytoplankton or zooplankton traits
'itera' implies a number, which stores the information of the different replica
combine_assembly_simulations.py
	Combines the simulation output from xxx
	Needs files assembly_mean_20_1000_'trait'_'itera'.npz
	assembly_var_20_1000_'trait'_'itera'.npz
	assembly_corr_20_1000_'trait'_'trait'_'itera'.npz
	Creates "files assembly_mean.csv" and "assembly_var.csv"
	"simplified_corr_phyto_traits_phyto.csv"
	"simplified_corr_phyto_traits_zoo.csv"
	"simplified_corr_zoo_traits_phyto.csv"
	"simplified_corr_zoo_traits_zoo.csv"
	"assembly_corr_zoo.csv",
	"assembly_corr_phyto.csv"
	"assembly_corr.csv" and "assembly_reference.csv"
combine_NFD_simulations.py
	Needs files "sim_NFD_mean_combined_'trait'_'itera'.npz"
	and "sim_NFD_corr_combined_'trait'_'trait'_'itera.npz"
	Creates "NFD_mean.csv", "NFD_corr.csv" and "NFD_ref.csv"
precombine_NFD_simulations.py
	Certain simulations were performed multiple times across different cores.
	This script combines these outputs into one file
	Needs the files "assembly_corr_1000_'trait'_'trait'_'itera'_'itera'.npz
	and sim_NFD_mean_1000_'trait'_'itera'_'itera'.npz
	Creates the files "sim_NFD_mean_combined_'trait'_'itera'.npz"
	and "sim_NFD_corr_'trait'_'trait'_'itera.npz"	

######################################################################
Plotting scripts:
layouts.py:
	Contains information to keep plotting consistent across figures
plot_ap_assembly_correlation_matrix.py
	Plots the figure "Figure_ap_assembly_correlation_matrix.pdf"
	Requires files "simplified_corr_phyto_traits_phyto.csv", "simplified_corr_phyto_traits_zoo.csv"
		"simplified_corr_zoo_traits_phyto.csv" and "simplified_corr_zoo_traits_zoo.csv"
plot_ap_assembly_var.py
	Plots the figure "Figure_ap_assembly_var.pdf"
	Requires files "assembly_var.csv"
plot_ap_base_case_analysis.py
	Plots the figure "Figure_ap_base_case_analysis.pdf"
	Requires file assembly_long_60_1000_0.npz
plot_ap_change_traits.pdf
	Plots the figure "Figure_ap_change_traits.pdf"
plot_ap_correlation_NFD_diversity.py
	Plots the figure "Figure_ap_correlation_NFD_diversity.pdf"
	Requires the file assembly_corr.csv
plot_ap_limiting_mechanism.py
	Plots the figure "Figure_ap_limiting_mechanism.py"
	Requires the files "assembly_non_limx_20_1000.npz
plot_ap_mean_NFD_diversity.py
	Plots the figure "Figure_ap_mean_NFD_diversity.pdf"
	Requires the file "assembly_mean.csv", "assembly_reference.csv"
	NFD_mean.csv and NFD_ref.csv
plot_assembly_correlation.py
	Plots the figure "Figure_assembly_correlation.pdf"
	Requires the files assembly_corr.csv and "assembly_reference.csv"
plot_assembly_mean.py
	Plots the figure "Figure_assembly_mean.pdf"
	Requires the files "assembly_mean.csv" and "assembly_reference.csv"
plot_NFD_on_richnes.py
	Plots the figures "Figure_NFD_on_richness.pdf", "Figure_ap_phyto_vs_zoo_richness.pdf"
	and "Figure_ap_correlation_NFD.pdf"
	Requires the files "assembly_mean.csv", "assembly_corr.csv"
	"NFD_mean.csv" and "NFD_corr.csv"
zoop_traits.py
	Plots the figure "Figure_zooplankton_traits.pdf"
	Requires the files "growth_rates_brun2017.csv", "clearance_rates_brun2017.csv",
	"Hirst_Kiorboe_2002.csv", "uye_1989.csv"
phyto_traits.py
	Plots the figure "Figure_phytoplankton_traits.pdf"
	Requires the files "Phytoplankton traits.csv", "Light_traits.csv"
	"augusti_data.csv" and "ehrlich2020.csv"