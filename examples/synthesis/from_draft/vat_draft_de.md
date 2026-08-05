# MwSt aufteilen

## Was es tun soll
Einen bruttobetrag inklusive Mehrwertsteuer in Netto und Steuer aufteilen.

## Eingaben
- brutto: Betrag inklusive MwSt, in Euro
- satz: MwSt-Satz als Bruch (z.B. 0,20 für 20%)

## Ausgabe
Ein Objekt mit netto und steuer in Euro.

## Regeln
- netto = brutto / (1 + satz)
- steuer = brutto - netto
- Beide auf zwei Nachkommastellen runden

## Beispiele
| brutto | satz | ergebnis |
|---|---|---|
| 120 | 0,20 | {"net": 100.0, "tax": 20.0} |
| 100 | 0 | {"net": 100.0, "tax": 0.0} |
| 1,00 | 0,20 | {"net": 0.83, "tax": 0.17} |
| 1.500,50 | 0,20 | {"net": 1250.42, "tax": 250.08} |
| 10 | 0,10 | {"net": 9.09, "tax": 0.91} |

## Worauf zu achten ist
Satz 0 bedeutet: alles ist Netto.
(Rundung an der halben Cent-Grenze ist nicht festgelegt.)

## Locale
de-DE
