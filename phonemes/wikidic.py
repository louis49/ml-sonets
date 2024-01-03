import xml.etree.ElementTree as ET
import re, json

# Chemin vers votre fichier XML
path_to_xml_file = 'frwiktionary-20231220-pages-articles-multistream.xml'

# Espace de noms défini dans le fichier XML
namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}  # ajoutez plus si nécessaire

langue_pattern = re.compile(r"== \{\{langue\|(.+?)\}\} ==")
prononciation_pattern = re.compile(r"\{\{pron\|[^}]*\}\}")


# Créer un itérateur pour événements de début et de fin
context = ET.iterparse(path_to_xml_file, events=('start', 'end'))

french_letters = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "É", "é",
    "È", "è", "À", "à", "Ù", "ù",
    "Â", "â", "Ê", "ê", "Î", "î", "Ô", "ô", "Û", "û",
    "Ë", "ë", "Ï", "ï", "Ü", "ü", "Ÿ", "ÿ",
    "Ç", "ç",
    "Œ", "œ", "Æ", "æ",
    # On est sympa avec les espagnols
    "Ñ", "ñ", "Ń", "ń",

    # L'espace n'est pas pas présent ainsi
]

french_phons = ["i", "e", "ɛ", "ɛ̃", "œ", "œ̃", "ə", "ø", "y", "u", "o", "ɔ",
                "ɔ̃", "ɑ̃", "ɑ", "a", "j", "ɥ", "w", "n", "ɲ", "ŋ", "ɡ", "k",
                "m", "b", "p", "v", "f", "d", "t", "ʒ", "ʃ", "z", "s", "ʁ",
                "l", "h", ".", '‿']


dic = {}
length = 0
for event, elem in context:
    if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
        # Extraire des informations de l'élément page
        title_elem = elem.find('mw:title', namespaces)
        ns_elem = elem.find('mw:ns', namespaces)
        id_elem = elem.find('mw:id', namespaces)

        # Afficher des informations
        if title_elem is not None and id_elem is not None and ns_elem.text == "0":
            text_elem = elem.find('.//mw:revision/mw:text', namespaces)
            lang_matches = langue_pattern.findall(text_elem.text)
            langue=''
            if lang_matches:
                langue = lang_matches[0]  # prenez le premier match comme exemple
            if langue == 'fr':
                prononciation_pattern = re.compile(rf"'''{re.escape(title_elem.text)}'''.*?\{{{{pron\|([^|}}]+)\|fr}}}}")
                pron_matches = prononciation_pattern.findall(text_elem.text)
                prononciation=''
                if pron_matches:
                    prononciation = (pron_matches[0]
                                     .replace('\n', '')
                                     .replace('(', '')
                                     .replace(')', '')
                                     .replace('/', '.')
                                     .replace('\'', '.')
                                     .replace('ˌ', '.')
                                     .replace('ʲ', '')
                                     .replace('ʀ', 'ʁ')
                                     .replace('r', 'ʁ')
                                     .replace('ɫ', 'l')
                                     .replace('ε', 'ɛ')
                                     .replace('ǝ', 'ə')
                                     )
                    titre = title_elem.text.replace('«', '').replace('а','a').replace('е', 'e') # Étrange notation pour la clef PCF

                    # On ne garde que les mots constitués de caractères français
                    if all(character in french_letters for character in titre) and all(character in french_phons for character in prononciation):
                        dic[titre] = {'phon': prononciation}
                        length += 1
                        if length % 10000 == 0:
                            print(length)
        elem.clear()


with open("dico.json", 'w', encoding='utf-8') as f:
    json.dump(dic, f, ensure_ascii=False, indent=4)
print(length)
# Nettoyer le parsing context
del context