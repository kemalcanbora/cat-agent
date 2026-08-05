"""Markdown draft templates for tool synthesis intake."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

# Headings only differ by language; structure is identical (D1 — prose is not parsed).

_EN = """\
# <tool name>

## What it should do
<One or two sentences, in your own words.>

## Inputs
- <name>: <what it means, which unit>

## Output
<What it should return.>

## Rules
<The business logic. Bullet points are fine.>

## Examples
| <input1> | <input2> | result |
|---|---|---|
|  |  |  |

## Things to watch out for
<Edge cases, exceptions. Leave blank if unsure — you will be asked.>

## Locale
<Optional. e.g. de-DE. Tells us how to read numbers and dates in your examples.>

<!--
Requirements:
- Provide at least 3 filled example rows.
- The last column is always the expected result.
- Leave anything unknown blank rather than guessing — you will be asked.
-->
"""

_DE = """\
# <Werkzeugname>

## Was es tun soll
<Ein oder zwei Sätze in Ihren eigenen Worten.>

## Eingaben
- <name>: <Bedeutung, Einheit>

## Ausgabe
<Was zurückgegeben werden soll.>

## Regeln
<Die Geschäftslogik. Aufzählungen sind in Ordnung.>

## Beispiele
| <eingabe1> | <eingabe2> | ergebnis |
|---|---|---|
|  |  |  |

## Worauf zu achten ist
<Grenzfälle, Ausnahmen. Leer lassen wenn unsicher — Sie werden gefragt.>

## Locale
<Optional. z.B. de-DE. Wie Zahlen und Daten in den Beispielen gelesen werden.>
"""

_FR = """\
# <nom de l'outil>

## Ce qu'il doit faire
<Une ou deux phrases, avec vos mots.>

## Entrées
- <nom>: <signification, unité>

## Sortie
<Ce qui doit être renvoyé.>

## Règles
<La logique métier. Les puces sont acceptées.>

## Exemples
| <entree1> | <entree2> | resultat |
|---|---|---|
|  |  |  |

## Points d'attention
<Cas limites, exceptions. Laissez vide en cas de doute — on vous demandera.>

## Locale
<Facultatif. ex. fr-FR. Comment lire les nombres et dates dans vos exemples.>
"""

_ES = """\
# <nombre de la herramienta>

## Qué debe hacer
<Una o dos frases, con sus propias palabras.>

## Entradas
- <nombre>: <significado, unidad>

## Salida
<Qué debe devolver.>

## Reglas
<La lógica de negocio. Viñetas están bien.>

## Ejemplos
| <entrada1> | <entrada2> | resultado |
|---|---|---|
|  |  |  |

## Cosas a tener en cuenta
<Casos límite, excepciones. Déjelo en blanco si no está seguro — se lo preguntaremos.>

## Locale
<Opcional. p.ej. es-ES. Cómo leer números y fechas en sus ejemplos.>
"""

_IT = """\
# <nome dello strumento>

## Cosa deve fare
<Una o due frasi, con parole sue.>

## Input
- <nome>: <significato, unità>

## Output
<Cosa deve restituire.>

## Regole
<La logica di business. Elenco puntato va bene.>

## Esempi
| <input1> | <input2> | risultato |
|---|---|---|
|  |  |  |

## Cose da tenere d'occhio
<Casi limite, eccezioni. Lasci vuoto se non sicuro — le verrà chiesto.>

## Locale
<Opzionale. es. it-IT. Come leggere numeri e date negli esempi.>
"""

_NL = """\
# <toolnaam>

## Wat het moet doen
<Eén of twee zinnen, in uw eigen woorden.>

## Invoer
- <naam>: <betekenis, eenheid>

## Uitvoer
<Wat het moet teruggeven.>

## Regels
<De bedrijfslogica. Opsommingen mogen.>

## Voorbeelden
| <invoer1> | <invoer2> | resultaat |
|---|---|---|
|  |  |  |

## Let op
<Randgevallen, uitzonderingen. Laat leeg als u twijfelt — u wordt gevraagd.>

## Locale
<Optioneel. bijv. nl-NL. Hoe getallen en datums in voorbeelden gelezen worden.>
"""

_TR = """\
# <araç adı>

## Ne yapmalı
<Bir veya iki cümle, kendi kelimelerinizle.>

## Girdiler
- <ad>: <anlamı, birimi>

## Çıktı
<Ne döndürmeli.>

## Kurallar
<İş kuralı. Madde işaretleri olabilir.>

## Örnekler
| <girdi1> | <girdi2> | sonuc |
|---|---|---|
|  |  |  |

## Dikkat edilecekler
<Kenar durumlar, istisnalar. Emin değilseniz boş bırakın — sorulacaktır.>

## Locale
<İsteğe bağlı. örn. tr-TR. Örneklerdeki sayı ve tarihlerin nasıl okunacağı.>
"""

TEMPLATES: Dict[str, str] = {
    'en': _EN,
    'de': _DE,
    'fr': _FR,
    'es': _ES,
    'it': _IT,
    'nl': _NL,
    'tr': _TR,
}


def get_template(lang: str = 'en') -> str:
    key = (lang or 'en').strip().lower().split('-')[0]
    return TEMPLATES.get(key, TEMPLATES['en'])


def write_template(path: Union[str, Path], lang: str = 'en') -> Path:
    """Write a blank draft template to *path* (UTF-8). Falls back to English."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(get_template(lang), encoding='utf-8')
    return out
