import xml.etree.ElementTree as ET

def sonnet_to_dic(sonnet):
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    title = sonnet.find('tei:head', namespaces=namespace).text.strip()
    strophes = []
    for strophe in sonnet.findall('tei:lg', namespaces=namespace):
        lines = [line.text.strip() for line in strophe.findall('tei:l', namespaces=namespace)]
        strophe = []
        for line in lines:
            strophe.append(line)
        strophes.append(strophe)

    return {'title': title, 'strophes': strophes}


def xml_to_dic():
    ET.register_namespace('', 'http://www.tei-c.org/ns/1.0')
    tree = ET.parse('../sonnets_oupoco_tei.xml')
    root = tree.getroot()
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    sonnets = root.findall(".//tei:div[@type='sonnet']", namespace)
    return [sonnet_to_dic(sonnet) for sonnet in sonnets]
