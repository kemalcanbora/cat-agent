# VAT split

## What it should do
Split a VAT-inclusive gross amount into net and tax given a rate.

## Inputs
- gross: amount including VAT, in euros
- rate: VAT rate as a fraction (for example 0.20 for 20%)

## Output
An object with net and tax amounts in euros.

## Rules
- net = gross / (1 + rate)
- tax = gross - net
- Round both to two decimal places

## Examples
| gross | rate | result |
|---|---|---|
| 120 | 0.20 | {"net": 100.0, "tax": 20.0} |
| 100 | 0.0 | {"net": 100.0, "tax": 0.0} |
| 1.00 | 0.20 | {"net": 0.83, "tax": 0.17} |
| 10 | 0.10 | {"net": 9.09, "tax": 0.91} |
| 250 | 0.18 | {"net": 211.86, "tax": 38.14} |

## Things to watch out for
Zero rate should return the whole amount as net.
(Rounding at the half-cent boundary is not specified — expect to be asked.)

## Locale
en-IE
