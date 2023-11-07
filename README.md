# Sintering densification prediction
This repository is part of my Bachelor Thesis. 
File `prediction.py` gets user process information and uses Gómez-Hotza(2018) model to predict the densification process during sintering

## Versions
**Version 1** - [`Model_GomezHotzaV1.ipynb`](https://github.com/gupaulasan/sintering-densification-prediction/blob/main/Model_GomezHotzaV1.ipynb) Uses a single experimental curve to obtain 3 different paramters by fitting the model to the curve. Comparisson is made with Mazaheri _et al._ (2009) experimental data.<br><br>
**Version 2** - [`Model_GomezHotzaV2.ipynb`](https://github.com/gupaulasan/sintering-densification-prediction/blob/main/Model_GomezHotzaV2.ipynb) Uses a three experimental curves to obtain 3 different paramters by fitting the model to the curve. Does it twice, in order to find the best fitting possible.  Comparisson is made with Duran _et al._ (1996) experimental data. <br><br>
**Version 3** - [`prediction.py`](https://github.com/gupaulasan/sintering-densification-prediction/blob/main/prediction.py) Uses the data obtained in Version 2 to create a Sintering Materials Database, in pandas. It has some user interaction features, such as selecting material and the process features. Returns a Matplotlib plot.

## References
All references are shown in the written work that will be [linked here]() when ready.
