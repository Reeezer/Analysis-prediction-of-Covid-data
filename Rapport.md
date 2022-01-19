# P3 - Analyse et prédiction de données Covid-19

------------------------------------------------------------------------------

Léon Muller - Projet P3 - 2021/2022 - Haute Ecole Arc - Projet n°221

Superviseur: M. Ninoslav

------------------------------------------------------------------------------

# Sommaire

------------------------------------------------------------------------------

1. Cahier des charges
	1. Récupération des données
	2. Visualisation des données
	3. Analyse des données et prédictions
2. Résumé
3. Introduction
	1. Récupération de données
	2. Visualisation des données
	3. Analyse de données et prédictions
	4. Bibliothèques utilisées
4. État de l'art
5. Planing
6. Récupération des données
7. Prétraitement
	1. Suppression des colonnes redondantes
	2. Analyse des NaN (Not a Number)
	3. Analyse de forme
	4. Analyse des corrélations
	5. Mise en forme
8. Prédictions
	1. Définitions
		1. Types d'algorithmes
		2. Types de problèmes
		3. Caractéristique d'un modèle
	2. Régression avec données
		1. Naive Bayes Classifier
			1. Gaussian
			1. Multinomial
		2. K-Nearest Neighbors
		3. Decision Trees
			1. Simple Decision Trees
			2. Random Forest
			3. Gradient Boosted Trees
		4. Support Vector Regression
		5. Multi-Layer Perceptrons (Deep Learning)
	3. Régression sans données
9. Conclusion
10. Bibliographie
	1. Sources documentaires
	2. Sources illustratives

------------------------------------------------------------------------------

# 1. Cahier des charges

------------------------------------------------------------------------------

L'intitulé du projet P3 est "Analysis and prediction of Covid-19 data using Python".
Ce projet a pour but de récolter des données sur le Covid (nombre de morts, de vaccinés, de lits de réanimation libres, etc ...) afin de les affichés de manière intuitive, compréhensive et utile. Pour ce faire, il sera nécessaire de former avec ces différentes données des schémas, diagrammes, cartes interactives, etc... pour permettre une visualisation la plus claire et pertinente possible. De plus, il faudra analyser ces différentes données afin de pouvoir en prédire les évolutions à venir.

Ce projet se découpe donc en trois grandes parties : récupération d'un set de données, visualisation des données récupérées, ainsi que prédictions des futurs évènements.

## 1.1 Récupération des données

- Recherche et choix de bases de données viables pour entrainer le modèle de machine learning, et libres d'accès
- Choix des données à traiter
- Mise en place d'un parseur pour traiter les données récoltées

Ex: our world in data

## 1.2 Visualisation des données

- Chercher, choisir et représenter les données sous la forme la plus adaptée et compréhensible possible (graphiques, diagrammes, ...)
- Choisir les différents paramètres pris en compte pour l'affichage des données (catégorie, temporalité, ...)
- Prise en mains des outils de visualisation tels que Pandas ou Seaborn

Voici quelques idées de données pertinentes à afficher :

- Couverture vaccinale (1e et 2nd dose)
- Nombre de tests positifs/négatifs (vaccinés ou non)
- Nombre d'hospitalisations
- Nombre d'admis en réanimation (vaccinés ou non)
- Nombre de nouveaux cas pour les mutations
- Taux de reproduction effectif (nombre de personnes contaminées par personne positive)
- Tension des reanimation
- Positifs, morts etc en cummulés
- Centres de dépistage sur une carte avec la position actuelle de l'ordinateur
- Carte des nombre de cas, lit de rea dispo, etc

Ce projet peut s'inspirer de sites internet ou applications déjà existantes permettant la visualisation de données liées au Covid tels que : https://www.gouvernement.fr/info-coronavirus/carte-et-donnees

Voici aussi quelques exemples de façons de représenter les données récoltées :

<img src="https://i.imgur.com/mGcibKB.png" height="200"/>
<img src="https://i.imgur.com/qUUrw4Q.png" height="200"/>
<img src="https://i.imgur.com/dO0Ysa3.png" height="200"/>
<img src="https://i.imgur.com/9sQqpWY.png" height="200"/>
<img src="https://i.imgur.com/VSm6qHf.png" height="200"/>

## 1.3 Analyse des données et prédictions

- Recherche et choix du/des modèle(s) utilisé(s) pour la prédiction des données
- Prise en mains des outils de machine learning tels que Tensorflow ou Keras
- Comparaison de différents modèles de machine learning afin d'en déterminer le/les meilleur(s)
- Test du/des modèle(s) de machine learning sélectionné(s) et réflexion sur leur efficacité
- Affichage des données prédites en rapport avec les données récoltées au préalable

------------------------------------------------------------------------------

# 2. Résumé

------------------------------------------------------------------------------

A faire en francais et anglais !

------------------------------------------------------------------------------

# 3. Introduction

------------------------------------------------------------------------------

Depuis le début de la crise sanitaire de Covid-19 en 2020, nous avons connu un grand changement dans nos vies quotidiennes. Les routines ont changé, ainsi que notre de travailler, ou encore d'interagir et même de nous comporter. Chaque jour, un grand nombre de données relatives au Covid-19 sont collectées sur le nombre de patients infectés, testés, décédés, guéris ou vaccinés. Les données sur les différentes mutations et les différents vaccins dans tous les pays du monde sont soigneusement collectées et pas toujours analysées ou visualisées de manière à pouvoir tirer certaines conclusions.

TODO agrandir l'introduction

Dans ce projet, l'objectif est donc de collecter et analyser les données Covid-19 pour différents pays et régions. Nous ferons des analyses statistiques et proposerons différentes méthodes de visualisation. En outre, nous essaierons de faire des prédictions en utilisant certains des outils et modules fournis par Python.

Ce projet se découpe donc en trois grandes parties : récupération de données, visualisation des données récupérées, ainsi que prédictions des futurs évènements.

------------------------------------------------------------------------------

## 3.1 Récupération de données

Au début du projet, il faudra chercher et choisir une base de données viable et libre d'accès pour entrainer le modèle de machine learning. Il faudra ensuite choisir les données à traiter, ainsi que mettre en place une manière de traiter les données récoltées.

------------------------------------------------------------------------------

## 3.2 Visualisation des données

Par la suite, il faudra chercher, choisir et représenter les données sous la forme la plus adaptée et compréhensible possible (graphiques, diagrammes ...). Par la suite, il sera nécessaire de choisir les différents paramètres pris en compte pour l'affichage des données (catégorie, temporalité ...). 
Pour cela, il faudra au préalable prendre en main des outils de visualisation tels que Pandas ou Seaborn.

------------------------------------------------------------------------------

## 3.3 Analyse de données et prédictions

Enfin, il faudra chercher, comparer, choisir et tester le/les modèle(s) de machine learning utilisé(s) permettant la prédiction du nombre de nouveaux morts chaque jour. Pour cela, il faudra au préalable prendre en main des outils de machine learning tels que Scikit-learn. Pour finir, le nombre de nouveaux morts chaque jour devra être affiché en rapport aux données récoltées au préalable.

------------------------------------------------------------------------------

## 3.4 Bibliothèques utilisées

------------------------------------------------------------------------------

Visualisation: 
- Matplotlib: Diagrammes simples et compréhensifs
- Seaborn: Diagrammes complexes et complets
- Plotly: Diagrammes interactifs

Représentation:
- Numpy: Représentation simpliste des données
- Pandas: Représentation des données sous forme de tableau labélisé

Machine learning:
- Scikit-learn: Modèles de machine learning, évaluation des modèles, mise en forme des données

------------------------------------------------------------------------------

# 4. État de l'art

------------------------------------------------------------------------------

TODO

État de l'art

Domaines dans lesquels on pourrait utiliser aussi ce projet

------------------------------------------------------------------------------

# 5. Planing

------------------------------------------------------------------------------

# 6. Récupération des données

------------------------------------------------------------------------------

Our World In Data est un site internet recensant un nombre impressionnant de données sur tout type de sujets à travers le monde. Les publications présentent sur le site sont dirigées par l'université d'Oxford et rédigées par l'historien social et économiste du développement Max Roser.

Ce site internet met à disposition une base de données accessible via GitHub: https://github.com/owid/covid-19-data/tree/master/public/data, ou directement sur leur site internet: https://ourworldindata.org/coronavirus.

Il y a à disposition plusieurs formats afin de récupérer cette base de données: csv, xlsx ou json. Json étant un format plus simple de compréhension, c'est celui que j'ai choisi de traiter. Cette base de données est tenue à jour et est actualisée chaque jour, ainsi les données traitées dans ce projet seront toujours actualisées.

Grâce à la bibliothèque Pandas, on peut récupérer les données en passant l'URL de recherche à la méthode `read_json`. 

------------------------------------------------------------------------------

Une fois les données récoltées, il va falloir les traiter afin d'en extraire les parties importantes et utiles. 

Dans le cadre de ce projet, nous allons traiter les données relatives à la Suisse, en utilisant donc l'ISO `CHE`. Les données Covid de chaque jour se trouvent ensuite dans la partie `data` de la base de données. 

On peut voir avec `tail`, que les données sont bien actualisées chaque jour. On peut aussi voir la forme des données, ainsi que les différentes informations contenues dans cette base de données. En effet, 45 données sont récoltées chaque jour en Suisse, ce nombre peut être différent en fonction du pays que l'on souhaite traiter.

------------------------------------------------------------------------------

Dans cette base de données, se trouvent aussi quelques informations à propos de chaque pays: le continent, la population, l'âge médian, le nombre de lits d'hôpitaux ... Toutes ces informations ne seront pas utilisées dans le cadre de ce projet, mais peuvent avoir une importance lors de la comparaison entre les données de différents pays.

------------------------------------------------------------------------------

# 7. Prétraitement

------------------------------------------------------------------------------

Dues à la grande taille de la base de données, ces données brutes sont généralement de faible qualité. Elles peuvent être incomplètes (valeurs manquantes), bruitées (valeurs erronées ou aberrantes) ou incohérentes (divergence entre attributs). Il est donc nécessaire d'effectuer un prétraitement sur ces données, d’améliorer la qualité des données.

Dans cette partie, on va donc modifier, travailler les données à disposition grâce à Pandas, afin de supprimer les parties inutilisées, d'uniformiser les données ... 

------------------------------------------------------------------------------

Étant donné que l'intégralité du projet se trouve dans ce notebook, il est vivement recommandé de ne jamais travailler directement sur le dataframe collecté, mais de passer par des copies pour pouvoir à tout moment récupérer les données brutes.

------------------------------------------------------------------------------

## 7.1 Suppression des colonnes redondantes

------------------------------------------------------------------------------

Dans la base de données se trouvent des colonnes redondantes, colonnes n'apportant aucune nouvelle information telles que le nombre de nouveaux cas par millions d'habitants, le nombre de cas lissé, ... Toutes ces colonnes ne seront pas utiles pour entrainer les modèles de machine learning, il faut donc les supprimer. ²

------------------------------------------------------------------------------

## 7.2 Analyse des NaN (Not a Number)

------------------------------------------------------------------------------

Dans notre base de données se trouvent des colonnes qui sont peu remplies (case remplie avec NaN), en effet les données collectées sur ces sujets n'ont pas été récupérées tous les jours, ou n'ont pas été collectées depuis le début de l'épidémie.

En effet, on peut voir sur le graphique ci-dessous le taux de données non répertoriées (en beige) pour chacune des colonnes de la base de données. Certaines données ne sont pas récoltées chaque jour (ex: weekly hosp admissions), et d'autres n'ont commencé à être récoltées qu'un certain temps après le début de la pandémie (ex: new vaccinations).

------------------------------------------------------------------------------

Voici aussi, sous forme textuelle, le schéma présent ci-dessus:

------------------------------------------------------------------------------

Les données étant collectées que depuis moins de 2 ans, la base de données mise à disposition est donc de faible taille. Il a donc fallu garder un maximum de colonnes, sans pour autant fausser les résultats pouvant se baser sur une partie trop importante de données manquantes. J'ai donc choisi de ne travailler que sur les colonnes contenant au moins 50% de données depuis le début de l'épidémie. Voici donc les différentes données restantes récoltées chaque jour disponibles après ce deuxième traitement:

------------------------------------------------------------------------------

## 7.3 Analyse de forme

------------------------------------------------------------------------------

Étant donné que le but du projet est de prédire le nombre de nouveaux morts par jour, nous allons travailler sur des modèles de machine learning basés sur la régression: prédire un nombre le plus proche de la réalité possible. Il est donc nécessaire de travailler sur des données sous formes numériques, ainsi vérifions le type de données présentes.

On doit tout de même garder la colonne `date` sous forme non numérique, car cette colonne deviendra par la suite l'indice du tableau.

------------------------------------------------------------------------------

La colonne `tests_units` n'est pas sous forme numérique, il faut donc la supprimer de nos données. Pour ce faire, on supprime donc toutes les colonnes qui sont sous la forme 'object', cependant cette méthode supprime aussi la colonne `date`, il faudra donc la rajouter par la suite, mais pour l'instant la supprimer aussi n'a pas d'importance. Voici donc les différentes données restantes récoltées chaque jour disponibles après ce troisième traitement:

------------------------------------------------------------------------------

## 7.4 Analyse des corrélations

------------------------------------------------------------------------------

Afin de supprimer les colonnes ne permettant pas d'aider le modèle de machine learning à mieux prédire le nombre de morts chaque jour, il est nécessaire d'effectuer une analyse de corrélation entre les différentes informations contenues dans la base de données. La corrélation mesure une dépendance linéaire entre deux variables. L'analyse de corrélation permet donc d’étudier la dépendance entre le nombre de morts chaque jour et les autres informations à disposition non supprimées jusqu'ici.

On peut représenter ces corrélations grâce à une matrice de corrélation.

------------------------------------------------------------------------------

Il est seulement nécessaire de se focaliser sur les corrélations en rapport avec ce qui sera prédit: `new_deaths`.

------------------------------------------------------------------------------

Plus le coefficient de corrélation est proche des valeurs extrêmes -1 et 1, plus la corrélation linéaire entre les variables est forte. On ne souhaite donc garder que les colonnes ayant une corrélation absolue avec `new_deaths` supérieure à 0.7.

![](https://i.imgur.com/VcJvg8V.png)

>http://www.sthda.com/french/wiki/test-de-correlation-formule

------------------------------------------------------------------------------

Voici donc les colonnes qui seront utilisées pour l'entraînement des modèles de machine learning.

------------------------------------------------------------------------------

## 7.5 Mise en forme

------------------------------------------------------------------------------

Une fois les colonnes utiles sélectionnées, il est nécessaire d'effectuer une mise en forme des données disponibles, car un certain nombre de données ne sont pas représentées, ou sont négatives or c'est tout simplement impossible.

On peut donc voir ci-après les différentes informations disponibles en fonction du temps qui seront utiles:

------------------------------------------------------------------------------

En observant attentivement le schéma du nombre de nouveaux morts chaque jour, il est possible d'apercevoir une saisonnalité, particularité que l'on ne retrouve que sur ce graphe. En effet, chaque week-end, le nombre de morts n'est pas forcément rempli, cela est flagrant entre octobre 2021 et janvier 2022: le nombre de morts chaque week-end est presque tout le temps nul. En revanche, j'en conclus que les données doivent être forcément reportées sur un jour suivant, par exemple le lundi ou le mardi, or ce n'est pas très visible sur ce graphe. 

TODO sure ?

Cette saisonnalité est un réel problème dans cette base de données, car à conditions égales et en fonction du jour de la semaine, les données rapportées ne seront pas les mêmes. J'ai donc choisi de directement supprimer l'intégralité des week-ends de la base de données afin de ne pas induire le modèle de machine learning en erreur.

D'autres approches auraient pu être: de rajouter le jour dans une nouvelle colonne de la base de données, mais cette solution n'est pas assez prise en compte par le modèle de machine learning. Ou encore de remplacer le nombre de morts du week-end par la moyenne pondérée des jours précédents et suivants, or le but est de prédire des données, il est n'est donc pas viable de n'utiliser seulement les données des jours précédents.

TODO pas ouf comme explication

------------------------------------------------------------------------------

On veut maintenant pouvoir remplir les données qui ne sont pas représentées. Pour ce faire, j'ai choisi de récupérer la valeur la plus proche pour chaque donnée manquante et remplir la base donnée avec ces informations. Cela permet de ne pas avoir de saut de données tout en gardant un maximum de données réelles pour entrainer le modèle de machine learning.

Une fois les données manquantes remplies, il faut supprimer les données négatives, car il est impossible qu'un nombre de morts soit négatif par exemple.

------------------------------------------------------------------------------

Le problème avec ce remplissage des données est que pour les colonnes `hosp_patients` et `icu_patients` ce remplissage n'est pas bien réalisé et ne peut pas être effectué simplement. Ces informations n'ont pas été récoltées depuis le début de la crise, et poseront problème au modèle. Pour palier à ce problème, j'ai choisir de supprimer l'intégralité des données récoltées avant le 30-03-2020.

------------------------------------------------------------------------------

Comme on peut le voir sur les graphes ci-dessus, une seule valeur est vraiment adhérente compte tenu des données adjacentes, il est donc nécessaire de la supprimer. Cependant, les données sont bien trop disparates, et il n'est donc pas possible de supprimer cette donnée automatiquement. La valeur que l'on retrouve le 09-02-2020 est donc retirée manuellement, permettant ainsi de lisser les données.

------------------------------------------------------------------------------

# 8. Prédictions

------------------------------------------------------------------------------

## 8.1 Définitions

------------------------------------------------------------------------------

Ces dernières années, les méthodes d'apprentissage automatique sont devenues omniprésentes dans la vie quotidienne. Des recommandations automatiques de films à regarder, de plats à commander ou de produits à acheter, la radio en ligne personnalisée ou la reconnaissance de vos amis sur vos photos, de nombreux sites Web et appareils modernes sont dotés d'algorithmes d'apprentissage automatique.

L'apprentissage automatique, aussi appelé "machine learning", est une sous-catégorie de l'intelligence artificielle qui est la capacité d'une machine à imité le comportement humain. L'intelligence artificielle est utilisée pour effectuer des tâches complexes d'une manière similaire à la façon dont les humains résolvent les problèmes.

------------------------------------------------------------------------------

### 8.1.1 Types d'algorithmes

------------------------------------------------------------------------------

Il existe deux types d'algorithmes d'apprentissage:
- Supervisé: algorithmes qui automatisent le processus de décision en généralisant à partir d'exemples connus. Dans ce cas, l'utilisateur fournit à l'algorithme des paires d'entrées et de sorties souhaitées, et l'algorithme trouve un moyen de produire la sortie souhaitée à partir d'une entrée. En effet, l'algorithme est capable de créer une sortie pour une entrée qu'il n'a jamais vue auparavant sans aucune aide.
- Non supervisés: dans ce type d'algorithmes, seules les données d'entrée sont connues, et aucune donnée de sortie connue n'est donnée à l'algorithme. 

Dans notre cas, comme les données à prédire sont connues, nous utiliserons des algorithmes d'apprentissage supervisé.

------------------------------------------------------------------------------

### 8.1.2 Types de problèmes

------------------------------------------------------------------------------

Dans ce type d'algorithmes, il existe deux types de problèmes d'apprentissage supervisé:
- Regression: l'objectif est de prédire un nombre continu. (Ex: prédire le prix d'une maison)
- Classification: l'objectif est de prédire une classe, qui est un choix parmi une liste prédéfinie de possibilités. (Ex: prédire la race d'un animal)

Dans notre cas, comme les données à prédire sont le nombre de nouveaux morts par jour, nous sommes dans un cas de régression.

------------------------------------------------------------------------------

### 8.1.3 Caractéristique d'un modèle

------------------------------------------------------------------------------

Dans l'apprentissage supervisé, nous voulons construire un modèle à partir de données, et être ensuite capables de faire des prédictions précises sur de nouvelles données. Si un modèle est capable de faire des prédictions précises sur des données inconnues, on dit qu'il est capable de généraliser de l'ensemble d'apprentissages à l'ensemble de tests. Nous voulons construire un modèle capable de généraliser aussi précisément que possible.

------------------------------------------------------------------------------

## 8.2 Régression avec données

------------------------------------------------------------------------------

Maintenant que les données récoltées ont été traitées pour être utilisées au mieux, il va nous être possible de les utiliser afin d'entrainer un modèle de machine learning et de pouvoir prédire des données que nous ne connaissons pas encore. Dans notre cas, nous utiliserons l'apprentissage automatique afin de pouvoir prédire le nombre de nouveaux morts chaque jour à cause du Covid-19. 

Dans le cadre de ce projet, la temporalité des données est une informations très importantes, nous allons donc travailler autour de cette caractéristique en posisitionnant la colonne `date` comme index du tableau de données.

------------------------------------------------------------------------------

**Division des données:**

Afin d'évaluer au mieux les performances d'un modèle de machine learning, il faut découper les données en trois grandes parties:
- les données d'entrainement (`train set`)
- les données de validation (`validation set`)
- les données de test (`test set`)

Pour commencer, il faut diviser en deux groupes les données de base: données de test, et données d'entrainement, en donnant un plus grande part aux données d'entrainement. Dans notre cas, il faut aussi prendre en compte la temporalité, il faut donc que les données d'entrainement soient au début, et les données de test à la fin. TODO 80%
Une fois les données divisées en deux catégories, il faut mettre de côté les données de test, ce sont les données qui nous permettront d'évaluer les différents modèles sur des données qu'ils ne connaissent pas.

<img src="https://i.imgur.com/JITVKcx.png" height="300">

Ensuite, pour évaluer un modèle lors de son entrainement, il est nécessaire de récupérer une partie des données restantes (données d'entrainement) afni de former les données de validation. Afin d'éviter tout biais dans l'apprentissage du modèle, il est nécessaire d'utiliser un `K-Fold cross validation` permettant de faire varier les données d'entrainement et les données de validation lors de l'apprentissage. Ce `K-Fold cross validation` divise en K parties les données d'entrainement, et forme une partie avec les données de validation, afin de tester toutes les combinaisons possibles d'entrainement.

<img src="https://i.imgur.com/VZrQkkD.png" height="300">

Cependant, dans ce projet, la temporalité des évènements importe énormément, il n'est donc pas viable d'utiliser un `K-Fold cross validation`, il faut plutot utiliser les `TimeSeriesSplit`. Cette classe permet d'effectuer de la même manière qu'un K-Fold, une séparation en K parties des données d'entrainement. Cependant, pour chaque entrainement du modèle, on utilise les données des entrainements précédents, ainsi que les 'nouvelles' données d'entrainement, le tout en fonction du temps.

<img src="https://i.imgur.com/ZDkrK1m.png" height="300">

**Evaluation des modèles:**

Après avoir entrainer les différents modèles de machine learning, il est nécessaire de les évaluer pour en déterminer les performances. Afin d'évaluer les différents modèles de machine learning, il faut déterminer une fonction d'évaluation qui sera utilisée par la suite pour évaluer tous les modèles. Pour ce faire, j'ai choisi la fonction `r2`, cette fonction représente le taux de corrélation des valeurs prédites avec les vraies données. Cette fonction d'évaluation a la particularité d'être normalisée, et d'être sous la forme d'un pourcentage, il est donc plus simple d'en interpréter les résultats. Cependant, il faut faire attention, car si les prédictions sont fortement anti-corrélées aux vraies données, le résultat sera aussi élevé.

**Comparaison avec une ligne de base:**

Pour voir si le modèle évalué est performant, il est nécessaire de le comparer avec une méthode dite 'naïve' de prédiction de données. En effet, si utiliser l'aléatoire est plus performant que de passer par le modèle de machine learning entrainé, c'est que le modèle n'est pas efficace. Comme dans ce projet, la temporalité des données est importante, j'ai choisi comme ligne de base de récupérer pour chaque jour le nombre de morts du jour précédent. Voici ci-dessous un exemple, avec les données représentés par des points orange, et la ligne de base décrite précédemment représentée par des traits orange.

<img src="https://i.imgur.com/fFW9qbd.png" height="300">

------------------------------------------------------------------------------

Pour chaque modèle de machine learning, les mêmes opérations vont être effectuées:
- création d'un `pipeline`
- définition des paramètres du modèle
- utilisation d'un `GridSearch` pour combiner le tout
- entrainement du modèle
- évaluation du modèle
- visualisation des résultats

**Création d'un `Pipeline`:**

TODO

**Définition des paramètres du modèle:**

Pour chaque modèle, il existe un certains nombre de paramètres pouvant être modifiés afin d'augmenter les performances du modèle en question. Comme l'ajustement de ces paramètres est une tâche très complexe, et dépend du cas que l'on souhaite traiter, j'ai utilisé une grille référencant les différents paramètres à modifier. Ainsi, toutes les combinaisons des paramètres seront testées afin d'en faire ressortir la meilleure par la méthode du `brut force`: test de toutes les combinaisons possibles.

**Utilisation d'un `GridSearch` pour combiner le tout:**

La classe `GridSearch` permet de combiner le `Pipeline` ainsi que la grille de paramètre, et d'autres fonctionnalités qui ne sont pas traitées dans le cadre de ce projet. Elle permet aussi de spécifier la fonction d'évaluation, la `cross validation`, et le nombre de coeurs de processeur utilisés. 

**Entrainement du modèle:**

Une fois la configuration terminée, il faut lancer l'entrainement des modèles sur les données d'entrainement.

**Evaluation du modèle:**

Afin d'évaluer les modèles, il est nécessaire de récolter le score de la fonction `r2` du modèle sur les données d'entrainement, les données de test, ainsi que sur la ligne de base pour pouvoir les comparer par la suite.

**Visualisation des résultats:**

Enfin, la visualisation des différentes données permet de mieux représenter les résultats obtenus précédemment. En effet, pour chaque modèle, un graphe est construit représentant les données d'entrainement, les données de test, l'entrainement du modèle, la prédiction du modèle, ainsi que la ligne de base. Ces graphes sont interactifs: il est possible de ne retirer l'affichage de certaines données en cliquant sur la légende, de zoomer sur les axes en sélectionnant une zone, etc ... En survolant les données il est aussi possible de voir le détail de celles-ci.

------------------------------------------------------------------------------

### 8.2.1 Naive Bayes Classifier

------------------------------------------------------------------------------

#### 8.2.1.1 Gaussian

------------------------------------------------------------------------------

#### 8.2.1.1 Multinomial

------------------------------------------------------------------------------

### 8.2.2 K-Nearest Neighbors

------------------------------------------------------------------------------

### 8.2.3 Decision Trees

------------------------------------------------------------------------------

#### 8.2.3.1 Simple Decision Trees

------------------------------------------------------------------------------

#### 8.2.3.2 Random Forest

------------------------------------------------------------------------------

#### 8.2.3.3 Gradient Boosted Trees

------------------------------------------------------------------------------

### 8.2.4 Support Vector Regression

------------------------------------------------------------------------------

### 8.2.5 Multi-Layer Perceptrons (Deep Learning)

------------------------------------------------------------------------------

## 8.3 Régression sans données

------------------------------------------------------------------------------

# 9. Conclusion

------------------------------------------------------------------------------

Knn pour prédire la forme des features, et ensuite utiliser ces features approximée pour approximer la feature à prédire



------------------------------------------------------------------------------

# 10. Bibliographie

------------------------------------------------------------------------------

## 10.1 Sources documentaires

------------------------------------------------------------------------------

## 10.2 Sources illustratives

------------------------------------------------------------------------------

Image KFold: https://www.researchgate.net/figure/The-technique-of-KFold-cross-validation-illustrated-here-for-the-case-K-4-involves_fig10_278826818

------------------------------------------------------------------------------

