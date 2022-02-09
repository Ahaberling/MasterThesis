# Knowledge Recombination and Diffusion in Patent Data - An Explorative Framework
MMDS Master Thesis 2021

* Objective: Exploring novel measurements concerned with knowledge diffusion and recombination by utilizing Latent Dirichlet Allocation and Network Analysis.  
* Data used: PATSTAT data provided by the Mannheimer Chair of Organization and Innovation.  
* Further details: *\'MasterThesis_Haberling.pdf\'*. 

## Instructions

The files contained in folder *\'Code\'* are expected to be exectued in the order indicated by their prefix.
The subfolder *\'utilities\'* contains files with custom functions, outsourced for the sake of readability.

The files with prefix 1-7 are based on Python 3.8
The file with prefix 8 is based on Python 3.7

The *cdlib* library utilized in *\'utilities/Measurement_utils.py\'* requires *Microsoft Visual C++ 14.0* or greater. (available at  
https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160)

The *Mallet* LDA utilized in *\'Code/2_Preprocessing_And_LDA.py\'* and *\'utilities/Data_Preparation_utils.py\'* requires a manual download (http://mallet.cs.umass.edu/download.php). Additionally a system variable named "MALLET_HOME" needs to point to the directory, in which the manual 
*Mallet* download was unpacked. At last the *Mallet* path variable in *\'Code/2_Preprocessing_And_LDA.py\'* need to be adjusted.

The *Gensim* library employed in *\'Code/2_Preprocessing_And_LDA.py\'* is versioned 3.8.3. Later versions do not support the employed Mallet wrapper.
