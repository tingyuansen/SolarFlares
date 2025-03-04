# Predicting Solar Flares with Bidirectional LSTM Networks  

Solar flares are harzards in space. Predicting solar flares would help prevent damages to space equipment and astronauts. Our goal
is to predict solar flares within the next 24 hours given a 12-hour time series of 25 magnetic parameters. This project is developed
for the [2019 BigDataCup Chanllenge](https://www.kaggle.com/c/bigdata2019-flare-prediction). The data sets for both training and test
can be downloaded [here](https://www.kaggle.com/c/bigdata2019-flare-prediction).

* `predicting_solar_flares.ipynb` is a jupyter notebook recording the whole developement process of this project, including the
flow of thought and how we tried different approaches before we choose the final approach.

* `predicting_solar_flares.py` is the python script version of the same code without the markdown cells.

The two files above only take use of a small part of the training data (`fold3Training.json` from the chanllenge) for the purpose
of model development. On the contrary,

* `train_pred_using_all_data.py` takes use of all three training data sets provided by the chanllenge, train an ensemble of models, and make predictions for the test data set.

* `read_json.py` contains functions for reading data from .json files and converting them to numpy arrays.

All three files above assume the data files are in the dir '../input'. If you put the data files elsewhere, please change the parameter `path_to_data` accordingly.

* `utils` is a directory contains codes in which we manually explore featue curation. This part is useful if you like to understand the physcial
parameters used in this project, otherwise you do not need to use anything from this directory.

## Dependencies

* tensorflow v2.0
* commonly used packages: pandas, numpy, matplotlib, scikit-learn
* I I develop this package in Python 3.7

## Authors

* [Jing Luan](https://sites.google.com/view/jingluan-astrophysics) -- jingluan.xw at gmail dot com
* [Sankalp Gilda](https://www.astro.ufl.edu/people/graduate-students/sankalp-gilda/) -- s.gilda at ufl dot edu
* [Yuan-Sen Ting](http://www.sns.ias.edu/~ting/) -- ting at ias dot edu

## Citing this code
* Please cite this repository, when using this code.

## Licensing

Copyright 2019 by Jing Luan.

In brief, you can use, distribute, and change this package as you please.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Acknowledgment

Jing Luan is supported by Association of Members of the Institute for Advanced Study (Amias) while developing this project.
