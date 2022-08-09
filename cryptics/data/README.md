## To create the data splits from scratch

1. Ensure the `data-annotated.sqlite3` database is in the same directory as the `create-splits.py` script
2. Ensure the `json/` directory has been created
3. Ensure the `json/naive-disjoint/`, `json/random/`, and `json/word-initial-disjoint/` directories have been created
4. Run `python3 create-splits.py`

## To alter the sizes of the splits

### Random Split

- Change the fraction used to set `val_start` and `test_start`

### Disjoint Splits

- The membership of a split is determined by the value of the hashed part of the answer. A modulus of the hash is taken with respect to the 'denominator' of the fraction representing the sizes of the splits. Later, the modulus of the hash is compared to the 'numerator' of the (cumulutive) fraction to determine the split it belongs to.
- This denominator will have to be changed, as well as the value against which the modulus of the hash is compared.
- E.g. for the 60/20/20 split, the (cumulutive) fraction for each split is $\frac{3}{5}/\frac{4}{5}/\frac{5}{5}$. $5$ is the simplified denominator, and the numerators represent the portion of data contained in previous splits _as well as_ the current split.

## Download

- ZIP files for the three splits are included in `json/` so that the splits do not have to be regenerated
