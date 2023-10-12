# Générateur de Sonets

Sur la base de [The Oupoco Database of French Sonnets from the 19th Century](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.89), ce dataset contient 4820 sonnets du 19ème siecle en Français.

## Contenu
* main.py : Entrée principale du code
* xml2json.py : Converti le fichier xml source en json pour une meilleure lisibilité
* json2text.py : Converti les données du sonnet en String en ajoutant les tags nécessaires à l'apprentissage et en nettoyant légèrement les données (caratères spéciaux, espaces etc...)
La forme de chaque sonnet est : 
```
<sonnet>
    <title> TITRE </title>  
    <stropheA> 
        <lineA> Ligne 1 <rime> r i </rime> mot qui rime </lineA> 
        <lineB> Ligne 2 <rime> r i </rime> mot qui rime </lineB> 
        <lineC> Ligne 3 <rime> r i </rime> mot qui rime </lineC> 
        <lineD> Ligne 4 <rime> r i </rime> mot qui rime </lineD> 
    </stropheA> 
    <stropheB> 
        <lineA> Ligne 1 <rime> r i </rime> mot qui rime </lineA> 
        <lineB> Ligne 2 <rime> r i </rime> mot qui rime </lineB> 
        <lineC> Ligne 3 <rime> r i </rime> mot qui rime </lineC> 
        <lineD> Ligne 4 <rime> r i </rime> mot qui rime </lineD> 
    </stropheB>
    <stropheC> 
        <lineA> Ligne 1 <rime> r i </rime> mot qui rime </lineA> 
        <lineB> Ligne 2 <rime> r i </rime> mot qui rime </lineB> 
        <lineC> Ligne 3 <rime> r i </rime> mot qui rime </lineC>  
    </stropheC> 
     <stropheD> 
        <lineA> Ligne 1 <rime> r i </rime> mot qui rime </lineA> 
        <lineB> Ligne 2 <rime> r i </rime> mot qui rime </lineB> 
        <lineC> Ligne 3 <rime> r i </rime> mot qui rime </lineC> 
    </stropheD>
</sonnet>
``` 
* text2seq.py : Converti les String du sonnet au format tf.Record
* sample_generator.py : Classe appelée pour afficher le résultat de la génération sur la base du titre : 
``` "<sonnet> <title> Amour fou </title>  <stropheA>  <lineA>"``` 

Le modèle est construit sur la base de [Tensorflow / Keras](https://keras.io)
Les hyper paramètres ont été définis en utilisant [keras_tuner](https://keras.io/keras_tuner/) 


# License

MIT

Exemple de sonnet généré après X Epochs : 
