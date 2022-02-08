## Knowledge Recombination and Diffusion in Patent Data
### An Explorative Framework

This thesis aimes to explore novel approaches identifying knowledge diffusion and recombination. The provided approaches utilize Latent Dirichlet Allocation and Network Analysis. Further details are provided in \'MasterThesis_Haberling.pdf\'. 


## Instructions

The files contained in folder \'Code\' are expected to exectued in the order indicated by their prefix.
The subfolder \'utilities\' contains files with custom functions, outsourced for the sake of readability.

The files with prefix 1-7 are based on Python 3.8
The file with prefix 8 is based on Python 3.7

The cdlib library utilized requires Microsoft Visual C++ 14.0 or greater.
Download it from 
https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160

Mallet LDA requires a manual download as well (http://mallet.cs.umass.edu/download.php).
In file 2_Preprocessing_And_LDA.py a Mallet path variable need to be adjusted.
Additionally a system variable named "MALLET_HOME" needs to point to the directory, in which the manual 
Mallet download was unpacked. 

The Gensim library employed in 2_Preprocessing_And_LDA.py is versioned 3.8.3. Later versions do not support the employed Mallet wrapper.
