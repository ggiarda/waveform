# Waveform

I want to recover the symbolic expression for the waveform of IMRPhenomA by using `PySR`. The functioanl form that I should recover isthe formula (4.16) present in [this article](https://arxiv.org/pdf/0710.2335.pdf).

The code is organized as follows:

* `Data Generation.ipynb` : this notebook generate the waveform and prepare the data for fitting.
* `fit.py` : this python scrypt does the fitting. It is a scrypt instead of a notebook because this makes it possibile to visualize the progress bar.
* `Data Analysis.ipynb` : this notebook is used to load the model obtained by fitting and analyse the result.
