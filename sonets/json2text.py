import re

import lirecouleur.word

SONNET_START = '<sonnet> '
SONNET_END = ' </sonnet> '
TITLE_START = '<title> '
TITLE_END = ' </title> '
STROPHE_START = ' <strophe{}> '
STROPHE_END = ' </strophe{}> '
LINE_START = ' <line{}> '
LINE_END = ' </line{}> '
RIME_START = '<rime> '
RIME_END = ' </rime>'

CHARS_TO_REMOVE= ['…', '‘', '|', '~', '`', '�', '■', 'µ', '[', ']', '(', ')', '\\', '/', '_', '-', '—',
                        '–', '»', '«', '"', '>', '<', '•', '1', '2', '3', '4', '5', '6', '7',
                        '8', '9', '0']

def number(num):
    if num == 0:
        return "A"
    elif num == 1:
        return "B"
    elif num == 2:
        return "C"
    elif num == 3:
        return "D"
    else:
        return "X"

def remove_parentheses(chaine):
    return re.sub(r'\(.*?\)', '', chaine).strip()

def remove_spaces(chaine):
    return re.sub(r'^\s+|\s+', ' ', chaine).strip()

def add_spaces(chaine):
    chaine = re.sub(r'(?<=[^\s])([?:.!,;])', r' \1 ', chaine)
    return re.sub(r'(?<=[^\s])([\'’])', r'\1 ', chaine)

def remove_quotes(chaine):
    return re.sub(r'^["\'](.*)["\']$', r'\1', chaine, flags=re.MULTILINE)

def clean(line):
    clean_line = line
    for char in CHARS_TO_REMOVE:
        clean_line = clean_line.replace(char, ' ')
    return clean_line

def add_rime(chaine):
    words = chaine.split()
    rime = ""
    index = len(words) - 1
    for index, mot in enumerate(reversed(words)):

        if len(mot) == 1:
            continue

        if mot == "faïencier":
            mot = "financier"
        if mot == "Hermès":
            mot = "kermesse"
        try:
            phen = lirecouleur.word.phonemes(mot)
            filtered_phen = [x[0] for x in phen if x[0] != "#"][-2:]
            rime = ' '.join(filtered_phen)
            break
        except Exception as e:  # Catche toutes les exceptions
            print(f"Une erreur est survenue : {e}")

    return (rime, index+1)

def dic2text(dic):
    sonnets = []
    for sonnet in dic:
        sonnet_block = SONNET_START
        sonnet_block += TITLE_START
        title = remove_parentheses(sonnet['title'])
        title = add_spaces(remove_spaces(clean(title)))
        if title == '':
            title = 'vide'
        sonnet_block += title
        sonnet_block += TITLE_END

        for i, strophe in enumerate(sonnet['strophes']):
            sonnet_block += STROPHE_START.format(number(i))
            for j, line in enumerate(strophe):
                sonnet_block += LINE_START.format(number(j))
                clean_line = remove_quotes(remove_spaces(add_spaces(clean(line))))
                rime, index = add_rime(clean_line)
                words = clean_line.split()
                words.insert(-index, RIME_START + rime + RIME_END)
                sonnet_block += ' '.join(words)

                sonnet_block += LINE_END.format(number(j))

            sonnet_block += STROPHE_END.format(number(i))

        sonnet_block += SONNET_END
        sonnets.append(sonnet_block)

    return sonnets