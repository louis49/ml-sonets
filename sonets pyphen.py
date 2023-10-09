import xml.etree.ElementTree as ET
import pyphen

import lirecouleur.text
import lirecouleur.word

#pyphen.language_fallback('fr_FR')
dic = pyphen.Pyphen(lang='fr_FR')
test = dic.inserted('fleur').split("-")[-1]
test3 = lirecouleur.text.syllables("Jusqu'au seuil de l'oubli m'a conduit par la main...".replace("!", " ").replace(";", " ").replace("-", " ").replace("—", " ")
                            .replace(",", " ").replace(".", " ").replace("'", " ").replace("?", " ")
                            .replace(":", " ")
                            .replace("»", " ").replace("«", " ").rstrip())

print(test)
def sonnet_to_dict(sonnet):
    # Obtenir le titre du sonnet
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

    sonnet_str = ET.tostring(sonnet, encoding='utf-8').decode('utf-8')
    title = sonnet.find('tei:head', namespaces=namespace).text.strip()

    # Obtenir les strophes et les lignes du sonnet
    strophes = []
    for lg in sonnet.findall('tei:lg', namespaces=namespace):
        lines = [line.text.strip() for line in lg.findall('tei:l', namespaces=namespace)]
        strophes.append(lines)

    return {'title': title, 'lines': strophes}

# Enlever les espaces de noms
ET.register_namespace('', 'http://www.tei-c.org/ns/1.0')

# Charger le XML depuis le fichier
tree = ET.parse('sonnets_oupoco_tei.xml')
root = tree.getroot()
namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
sonnets = root.findall(".//tei:div[@type='sonnet']", namespace)

# Convertir chaque élément sonnet en dictionnaire
sonnets_dicts = [sonnet_to_dict(sonnet) for sonnet in sonnets][:3000] #2000

# Prétraitement des données
titles = [sonnet["title"].replace(".", "") for sonnet in sonnets_dicts]

lengths = []
sonnets_blocks = []
for sonnet in sonnets_dicts:
    sonnets_block = " <start> "
    for strophe in sonnet["lines"]:
        sonnets_block += " <strophe> "
        for line in strophe:
            sonnets_block += " <ligne> "

            words = line.split()
            if line.find("Hermès") > -1:
                phen2 = "mès"
            elif line.find("faïencier") > -1:
                phen2 = "cier"
            else:
                phen2 = lirecouleur.text.syllables(line.replace("!", " ").replace(";", " ").replace("-", " ").replace("—", " ")
                            .replace(",", " ").replace(".", " ").replace("'", " ").replace("?", " ")
                                                   .replace(":", " ").replace("»", " ").replace("«", " ").rstrip()
                                                   )

                if len(phen2) == 0 :
                    print()
                if len(phen2[-1]) == 0 :
                    print()
                phen2 = phen2[-1][-1]


            if phen2 == " ":
                print("")

            if phen2 == "":
                print("")

            phen3 = lirecouleur.word.phonemes(
                line.replace("!", " ").replace(";", " ").replace("-", " ").replace("—", " ")
                .replace(",", " ").replace(".", " ").replace("'", " ").replace("?", " ")
                .replace(":", " ").replace("»", " ").replace("«", " ").rstrip().split()[-1])

            if len(words) > 2:
                if(len(words[len(words)-1]) == 1):
                    word = (words[-2].replace("!", "").replace(";", "")
                            .replace(",", "").replace(".", "").replace("'", "")
                            .replace("?", ""))

                    phen = dic.inserted(word).split("-")[-1]
                    print(phen, phen2, phen3)
                    words.insert(-2, "<rime-" + phen + ">")
                else:
                    word = (words[-1].replace("!", "").replace(";", "")
                            .replace(",", "").replace(".", "").replace("'", "").replace("?", ""))
                    phen = dic.inserted(word).split("-")[-1]
                    print(phen, phen2, phen3)
                    words.insert(-1, "<rime-" + phen + ">")

            # Reconstruct the line with the tag
            line = ' '.join(words)

            sonnets_block += line.replace(".", " . ").replace(",", " , ").replace("!", " ! ")
    sonnets_block += " <end> "
    sonnets_blocks.append(sonnets_block)
    lengths.append(len(sonnets_block))