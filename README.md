# RAMP starting-kit on Purchasing Intention prediction for online shoppers


Authors : Aymane Rahmoune, Haocheng LIU, Ly An CHHAY, Mohammed Jawhar, Nasr El Hamzaoui, Wiam Adnan

## Getting started

### Install

To run a submission and the notebook you need to install the dependencies listed
in `requirements.txt`. You can do this with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [shopper_intention_starting_kit](shopper_intention_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

## Important Note

If you encounter an error during the execution of this command, please do the following steps:

* Find out the local environment in which ramp-test is installed using which ramp-test. This will give you something that ends with bin/ramp-test, copy the path before.
* Go to the ramp classifier workflow file, which corresponds to the file at the path obtained by adding /lib/python3.9/site-packages/rampwf/workflows/classifier.py at the end of the path you got just before.
* In this file, replace:
  ```python
  if prev_trained_model is None:
     clf.fit(X_array[train_is], y_array[train_is])
  else:
     clf.fit(X_array[train_is], y_array[train_is], prev_trained_model)
  return clf
  ```
  
  By:
  ```python
  import pandas as pd
  if prev_trained_model is None:
     if isinstance(X_array, pd.core.frame.DataFrame):
        clf.fit(X_array.iloc[train_is], y_array[train_is])
     else:
        clf.fit(X_array[train_is], y_array[train_is])
  else:
     if isinstance(X_array, pd.core.frame.DataFrame):
        clf.fit(
            X_array.iloc[train_is],
            y_array[train_is],
            prev_trained_model,
        )
      else:
        clf.fit(
            X_array[train_is], y_array[train_is], prev_trained_model
        )
  return clf
  ```

This should solve the problem, allowing the command to run successfully.

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

