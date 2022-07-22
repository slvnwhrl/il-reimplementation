# A neural transducer
This package contains a cli-based neural transducer for string transduction tasks. The transducer builds on the model by Makarov & Clematide (2020). This implementation introduces GPU-supported mini-batch training and batched greedy decoding as well as support for transformer-based encoders (see Wehrli & et al. (2022) for further information).

## Installation
Please make sure that you are using Python 3.7.
To install this package, perform the following steps:

* Clone the development branch and change to the package directory:

        git clone --single-branch -b development https://github.com/slvnwhrl/il-reimplementation.git neural_transducer
        cd neural_transducer

* Install the package:

  * default installation

        pip install .
  
  * with cuda support
        
        pip install --cuda

  * local development (without the need to reinstall the package after changes):

        pip install -e .

* Optionally, run unit tests:

        python setup.py test

## Usage
### Training
In order to train a model, directly run the python script ``train.py`` 
via ``python train.py`` or use the cli entry point ``trans-train``.

The most important (and required) parameters are:
* ``--train`` path to the training data
* ``--dev`` path to the development data
* ``--output`` path to the output directory

For a full list of available training configurations, use ``trans-train --help``.

### Ensembling
To ensemble a number of models based on majority voting, run the python script 
``ensembling.py`` via ``python ensembling.py`` or use the cli entry point 
``trans-ensemble``. The following parameters are required:
* ``--gold`` path to the gold data
* ``--systems`` path to the systems' data
* ``--output`` path to the output directory

### Grid Search
In order to enable efficient model (hyper)parameter exploration,
this package offers grid search that allows to run a defined number of models
for specified configurations. To specify configurations, 
a JSON file is used (see below for further explanations).
To run grid search, run the python script ``grid_search.py`` via 
``python grid_search.py`` or use the cli entry point ``trans-grid-search``. 

The following parameters are available:
* ``--config`` path to the JSON config file (required)
* ``--ouput`` path to the output directory (required)
* ``--parallel-jobs`` number of jobs (i.e., trainings) that are run in parallel (on CPU and GPU)
* ``--ensemble`` bool indicating whether to produce ensemble results or not

The command ``trans-grid-search --help`` can be run to get information about 
the available parameters.

#### Configuration file
The JSON-based configuration file needs to be passed via ``--config`` parameter.
It basically contains information about the used data as well as model (hyper)parameters.
An [example](trans/docs/grid_search_config_example.json) can be found in the docs folder. The schema for the JSON file is
defined as following:

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Grid search config",
  "description": "Configuration file for grid-based search/optimization",
  "type": "object",
  "properties": {
    "data": {
      "description": "Information about the used data",
      "type": "object",
      "properties": {
        "path": {
          "description": "The path to the training data",
          "type": "string"
        },
        "pattern": {
          "description": "The pattern used to find the split- and language-specific data files. In the given path the words SPLIT and LANG will be replaced.",
          "type": "string"
        },
        "languages": {
          "description": "List of languages to train on",
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    },
    "required": [
      "path",
      "pattern",
      "languages"
    ]
  },
  "runs_per_model": {
    "description": "The number of models that are trained per possible combination of config parameters.",
    "type": "integer"
  },
  "grids": {
    "description": "This objects contains specification for named grids.",
    "type": "object",
    "patternProperties": {
      "^.*$": {
        "description": "This object represents a single grid and contains the model (hyper)parameters as key-value pairs. Note that values can either be passed as single values or as array of different values. An exception is the sed-params parameter which expects a object containing key-value pairs of languages and paths.",
        "type": "object"
      }
    }
  },
  "required": [
    "data",
    "runs_per_model",
    "grids"
  ]
}
```

In principle, all parameter values can either be passed as a single value or 
as an array of values. In any case, all possible combinations of all passed
parameter values for a specific grid will be produced and used for training. However,
two things should be noted:
* Firstly, for parameters without required values (e.g., ``--nfd``) a boolean needs
to be specified.
* Secondly, the model parameter ``--sed-params`` expects a dictionary that contains
key value pairs of language name and path to the sed parameters. If a key for a
specific language is missing, a new sed aligner will be trained.

#### Output structure
All output will be generated in the folder specified by the ``--output`` cli argument.
This folder contains a separate folder for each grid that is specified in the config folder.
The name is defined by the name used as key values in the ``grids`` property 
of the config file. This folder contains a `combinations.json` file that 
describes the different possible combinations and maps each combination to a number.
Additionally, this folder contains a separate folder for each trained language.
Each of these "language folders" contains a folder for each possible grid combination
(--> number from `combinations.json`) which, in turn, contain all the trained
models for this specific configuration. Additionally, a results text file is produced
that documents the performance average (accuracy) of all runs. If the ``--ensemble``
parameter is passed, separate results text files will be produced.

#### References
P. Makarov and S. Clematide. [CLUZH at SIGMORPHON 2020 Shared Task on Multilingual Grapheme-to-Phoneme Conversion](https://aclanthology.org/2020.sigmorphon-1.19). In *Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*, 2020.

S. Wehrli, S. Clematide, and P. Makarov. [CLUZH at SIGMORPHON 2022 Shared Tasks on Morpheme Segmentation and Inflection Generation](https://aclanthology.org/2022.sigmorphon-1.21). In *19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*, 2022.
