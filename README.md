# RaVeN
RaVeN : Relational Verification of Neural Networks

Relational erifier runs through the unittest framework. A new unit test can be added to run the verifier
with a specific configuration. 
Current unit tests are located in `src/tests/test_raven.py`. 
Conda Environment at `environment.yml`, install with `conda env create -f environment.yml`.

## Reproducing Experiments

Move to the RaVeN directory

```
cd RaVeN
```

Then run the following any of the following commands 

### Untargeted UAP Verification

Run any experiment by replacing ``test_name`` with any test from class ``TestDifferenceAblation`` from the file ``RaVeN/src/tests/test_raven.py``

Small networks
```
python -m unittest -v src.tests.test_raven.TestUntargetedUapSmall.test_name
```

Large networks
```
python -m unittest -v src.tests.test_raven.TestUntargetedUapLarge.test_name

```

### Targeted UAP Verification

Run any experiment by replacing ``test_name`` with any test from class ``TestTargetedUap`` from the file ``RaVeN/src/tests/test_raven.py``
```
python -m unittest -v src.tests.test_raven.TestTargetedUap.test_name
```

### Monotonicity Verification

Run any experiment by replacing ``test_name`` with any test from class ``TestMonotonicity`` from the file ``RaVeN/src/tests/test_raven.py``
```
python -m unittest -v src.tests.test_raven.TestMonotonicity.test_name
```

### Worst Worst-case hamming distance Verification 

Run any experiment by replacing ``test_name`` with any test from class  ``TestWorstCaseHamming`` from the file ``RaVeN/src/tests/test_raven.py`
```
python -m unittest -v src.tests.test_raven.TestWorstCaseHamming.test_name
```


### Ablation w.r.t Difference Constraints
Run any experiment by replacing ``test_name`` with any test from class  ``TestDifferenceAblation`` from the file ``RaVeN/src/tests/test_raven.py``

python -m unittest -v src.tests.test_raven.TestDifferenceAblation.test_name


## Designing Custom Tests

1. Update the parameters in the ``test_custom`` from ``TestRavenCustom`` in the file ``RaVeN/src/tests/test_raven.py``.
The class ``RaVeNArgs`` from ``RaVeN/src/raven_args.py`` defines the parameters

2. After updating parameters Move to the RaVeN directory
```
cd RaVeN
```

3. Run the following command from
```
python -m unittest -v src.tests.test_raven.TestRavenCustom.test_custom
```
