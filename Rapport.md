# P3 - Analyse et prédiction de données Covid-19

------------------------------------------------------------------------------

Léon Muller - Projet P3 - 2021/2022 - Haute Ecole Arc - Projet n°221

Superviseur: M. Ninoslav

------------------------------------------------------------------------------

# Sommaire

------------------------------------------------------------------------------

1. Introduction
2. Cahier des charges
	1. Récupération des données
	2. Visualisation des données
	3. Analyse des données et prédictions
3. Résumé
	1. Version française
	2. Version anglaise 
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

# 1. Introduction

------------------------------------------------------------------------------

Depuis le début de la crise sanitaire de Covid-19 en 2020, nous avons connu un grand changement dans nos vies quotidiennes. Les routines ont changé, ainsi que notre façon de travailler, ou encore de nous comporter. Depuis presque 2 ans, cette maladie touche une grande quantité de personnes provoquant ainsi chez elles des symptomes allant des plus bénins aux plus mortels. Cette épidémie contraint les états à renforcer les campagnes de santé, appliquer des règles nouvelles, ainsi que forcer des populations entières au confinement. 

La Covid-19 est un virus qui mute au fil du temps, certaines mutations sont plus ou moins dangereuses et plus ou moins transmissibles. Ce qui rend ce virus imprévisible et compliqué à soigner complètement. Il n'existe encore aucun traitement à ce jour permettant de guérir et d'éradiquer cette maladie, il est donc primordial pour nous être Humain d'apprendre à vivre avec. 

Depuis le début de cette pandémie, dans presque tous les pays du monde, un grand nombre de données relatives au Covid-19 sont collectées. Ces données recensent le nombre de patients infectés, testés, guéris, le nombre de lits de réanimation libre, ou encore le taux de saturation des hopitaux, etc. Afin de permettre aux populations, mais aussi aux gouvernements de mieux se rendre compte de la situation actuelle, certaines solutions ont été mises en place telles que l'affichage sous forme de graphique des différentes données, le calcul de l'évolution semaine après semaine de la pandémie, ou encore l'affichage sur carte des zones les plus touchées. En effet, ces différentes représentations permettent de mieux comprendre et visualiser l'état actuel de la situation et de mieux comprendre la raison des décisions gouvernementales par exemple.

Enfin, grâce à la quantité de données récoltée, certains organismes tels que l'Institute for Health Metrics and Evaluation mettent en place des formes d'intelligence artificielle visant à prévoir l'évolution de la situation pandémique dans les semaines ou mois à venir. Ces solutions sont d'ores et déjà diverses et variées et n'apportent pas toutes les mêmes conclusion quant aux évolutions sanitaires. 

C'est pourquoi, dans ce projet l'objectif est de collecter et analyser les données Covid-19 pour différents pays et régions. Nous ferons des analyses statistiques et proposerons différentes méthodes de visualisation. En outre, nous essaierons de faire des prédictions et comparer les solutions trouvées à celles déjà existantes pour en montrer ou non la pertinence. 

Ce projet se découpe donc en trois grandes parties : récupération de données, visualisation des données récupérées, ainsi que prédictions des futurs évènements.

------------------------------------------------------------------------------

**Récupération de données**

Au début du projet, il faudra chercher et choisir une base de données viable et libre d'accès pour entrainer le modèle de machine learning. Il faudra ensuite choisir les données à traiter, ainsi que mettre en place une manière de traiter les données récoltées.

------------------------------------------------------------------------------

**Visualisation des données**

Par la suite, il faudra chercher, choisir et représenter les données sous la forme la plus adaptée et compréhensible possible (graphiques, diagrammes ...). Par la suite, il sera nécessaire de choisir les différents paramètres pris en compte pour l'affichage des données (catégorie, temporalité ...). 
Pour cela, il faudra au préalable prendre en main des outils de visualisation tels que Pandas ou Seaborn.

------------------------------------------------------------------------------

**Analyse de données et prédictions**

Enfin, il faudra chercher, comparer, choisir et tester le/les modèle(s) d'apprentissage automatique utilisé(s) permettant la prédiction du nombre de nouveaux morts chaque jour. Pour cela, il faudra au préalable prendre en main des outils de machine learning tels que Scikit-learn. Pour finir, le nombre de nouveaux morts chaque jour devra être affiché en rapport aux données récoltées au préalable.

------------------------------------------------------------------------------

**A noter**

Ce notebook inclu rapport et code, chaque partie est donc détaillée afin d'expliquer au mieux les différents scipts réalisées. En effet, chaque morceau de code est expliqué et est intéractif, il est donc possible de lancer le notebook afin d'observer les résultats en temps réel. Cependant, certains scripts ont besoin de beaucoup de temps pour s'exécuter, il est donc déconseillé de lancer l'intégralité du notebook d'un coup. Les cases prenant le plus de temps à s'exécuter sont celles entrainant les modèles d'apprentissage automatique complexes.

------------------------------------------------------------------------------

# 2. Cahier des charges

------------------------------------------------------------------------------

L'intitulé du projet P3 est "Analysis and prediction of Covid-19 data using Python".
Ce projet a pour but de récolter des données sur le Covid (nombre de morts, de vaccinés, de lits de réanimation libres, etc ...) afin de les affichés de manière intuitive, compréhensive et utile. Pour ce faire, il sera nécessaire de former avec ces différentes données des schémas, diagrammes, cartes interactives, etc... pour permettre une visualisation la plus claire et pertinente possible. De plus, il faudra analyser ces différentes données afin de pouvoir en prédire les évolutions à venir.

Ce projet se découpe donc en trois grandes parties : récupération d'un set de données, visualisation des données récupérées, ainsi que prédictions des futurs évènements.

## 2.1 Récupération des données

- Recherche et choix de bases de données viables pour entrainer le modèle de machine learning, et libres d'accès
- Choix des données à traiter
- Mise en place d'un parseur pour traiter les données récoltées

Ex: our world in data

## 2.2 Visualisation des données

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

> Différentes captures d'écran provenant d'applications ou de site web (TousAntiCovid, CovidTracker, SwissInfo, ...) mettant en avant la situation du Covid-19 

## 2.3 Analyse des données et prédictions

- Recherche et choix du/des modèle(s) utilisé(s) pour la prédiction des données
- Prise en mains des outils de machine learning tels que Tensorflow ou Keras
- Comparaison de différents modèles de machine learning afin d'en déterminer le/les meilleur(s)
- Test du/des modèle(s) de machine learning sélectionné(s) et réflexion sur leur efficacité
- Affichage des données prédites en rapport avec les données récoltées au préalable

------------------------------------------------------------------------------

# 3. Résumé

------------------------------------------------------------------------------

## 3.1 Version française

------------------------------------------------------------------------------

Ce projet a pour sujet "Analysis and prediction of Covid-19 data using Python". Le projet a pour but d'utiliser une base de données libre d'accès et mise à jour quotidiennement afin d'en extraire des données utiles, et ainsi tenter de prédire certaines données avant qu'elles ne soient révélées. Pour ce projet a été choisi comme donnée de prédiction, le nombre de mort du Covid chaque jour. L'objectif est donc de prédire cette information grâce aux autres données disponibles dans la base de données récupérée. 

Dans un premier temps, une base de données provenant de 'Our world in data' a été choisie. Cette base de données recense chaque jour des milliers d'informations liées au Covid-19 dans près de 250 pays du monde. Il est ensuite nécessaire de filtrer ces données pour n'en garder que les plus pertinentes. Pour ce faire, différentes analyses ont été réalisées permettant de supprimer les données dont la quantité récoltée est trop faible, mais aussi de ne garder que les données ayant un lien avec le nombre de morts par jour, ou encore les données redondantes telles que le nombre de cas positif au Covid par milliers, par millions, etc. De plus, dans le cadre de ce projet, seules les données provenant de la Suisse ont été utilisées.

Une fois les données filtrées et analysées, elles ont été utilisées afin d'entrainer des algorithmes d'apprentissage automatique. Plusieurs algorithmes ont été entrainés pour pouvoir les comparer, comprendre leur fonctionnement, les optimiser et ainsi trouver celui ayant les meilleures performances, ce qui donc permet de prédire au mieux le nombre de morts du Covid chaque jour. Les différentes prédictions et données récoltées chaque jour ont été représentées sur des graphiques permettant visuellement de comprendre les résultats obtenus. Il est ainsi possible de voir comment se comporte chaque algorithme, et d'en faire des déductions.

Cependant, cette méthode ne permet que de prédire par exemple le nombre de morts du jour actuel en utilisant les données de ce jour. J'ai donc tenté d'aller plus loin, et de prédire le nombre de morts pour la ou les semaines à venir, mais évidemment sans données. Cette tentative soulève des questions et amène à des améliorations quant à la méthode et justesse de la prédiction de données dans le futur.

------------------------------------------------------------------------------

## 3.2 Version anglaise 

------------------------------------------------------------------------------

This project is about "Analysis and prediction of Covid-19 data using Python". The aim of the project is to use an open source database that is updated daily to extract useful data, and thus attempt to predict certain data before they are revealed. For this project, the number of Covid deaths per day was chosen as the predictive data. The objective is to predict this information thanks to the other data available in the recovered database. 

In a first step, a database from 'Our world in data' was chosen. This database lists thousands of pieces of information related to Covid-19 in nearly 250 countries around the world every day. It is then necessary to filter this data to keep only the most relevant. To do this, various analyses were carried out to remove data whose quantity collected is too small, but also to keep only data related to the number of deaths per day, or redundant data such as the number of Covid-positive cases per thousand, per million, etc. In addition, only data from Switzerland was used in this project.

Once the data was filtered and analysed, it was used to train machine learning algorithms. Several algorithms were trained in order to compare them, understand how they work, optimise them and find the one with the best performance, which therefore allows the best prediction of the number of Covid deaths each day. The different predictions and data collected each day have been represented on graphs allowing a visual understanding of the results obtained. It is thus possible to see how each algorithm behaves, and to make inferences.

However, this method can only predict, for example, the number of deaths on the current day using the data for that day. I have therefore tried to go further and predict the number of deaths for the coming week(s), but obviously without data. This attempt raises questions and leads to improvements in the method and accuracy of predicting data in the future.

------------------------------------------------------------------------------

# 4. État de l'art

------------------------------------------------------------------------------

TODO

État de l'art

Domaines dans lesquels on pourrait utiliser aussi ce projet

------------------------------------------------------------------------------

# 5. Planing

------------------------------------------------------------------------------

Au début du projet, un planing a été réalisé afin de diviser le projet en plusieurs tâches, et ainsi permettant de voir l'avance ou le retard pris régulièrement en fonction des objectifs fixés. Les différentes tâches ont été regroupées en troids grandes parties: récupération des données, prédiction, visualisation.

<img src="https://i.imgur.com/J7oYvJ3.png" height="500">

Le planing a été en majorité respecté. Cependant, les étapes de visualisation des données ont été réalisées en parallèle des étapes de prédiction et de prise en main des modèles de machine learning. Il était nécessaire de pouvoir constater visuellement les données résultant des prédictions des différents modèles. De plus, le choix de base de données et donc la partie traitement des données a été bien plus longue que prévue initialement.

Une adaptation du planing a donc du être réalisée, et les écarts engendrés ont pu être rattrapés avec une charge de travail plus importantes durant certaines semaines. 

------------------------------------------------------------------------------

# 6. Récupération des données

------------------------------------------------------------------------------

Avant toute chose, afin de pouvoir exécuter et tester les différents scripts présents dans ce notebook, il faut importer et installer les bibliothèques utilisées. Pour ce faire, il suffit, avec `pip` installé, de lancer la commande suivante (dans un terminal à la racine du dossier dans lequel se trouve ce notebook): `pip install -r requirements.txt`. Cette commande va installer automatiquement les versions des bibliothèques spécifiées dans le fichier `requirements.txt`.

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

Our World In Data est un site internet recensant un nombre impressionnant de données sur tout type de sujets à travers le monde. Les publications présentent sur le site sont dirigées par l'université d'Oxford et rédigées par l'historien social et économiste du développement Max Roser.

Ce site internet met à disposition une base de données accessible via GitHub: https://github.com/owid/covid-19-data/tree/master/public/data, ou directement sur leur site internet: https://ourworldindata.org/coronavirus.

Il y a à disposition plusieurs formats afin de récupérer cette base de données: csv, xlsx ou json. Json étant un format plus simple de compréhension, c'est celui qui a été choisit. Cette base de données est tenue à jour et est actualisée quotidiennement, ainsi les données traitées dans ce projet seront toujours actualisées.

Grâce à la bibliothèque Pandas, on peut récupérer les données en passant l'URL de recherche à la méthode `read_json`. 

------------------------------------------------------------------------------

Une fois les données récoltées, il va falloir les traiter afin d'en extraire les parties importantes et utiles. 

Dans le cadre de ce projet, nous allons traiter les données relatives à la Suisse, en utilisant donc l'ISO `CHE`. Les données Covid de chaque jour se trouvent ensuite dans la partie `data` de la base de données. 

On peut voir avec `tail`, que les données sont bien actualisées chaque jour. On peut aussi voir la forme des données, ainsi que les différentes informations contenues dans cette base de données. En effet, 45 données différentes sont récoltées chaque jour en Suisse, ce nombre peut varier en fonction du pays que l'on souhaite traiter.

------------------------------------------------------------------------------

Dans cette base de données, se trouvent aussi quelques informations à propos de chaque pays: le continent, la population, l'âge médian, le nombre de lits d'hôpitaux, etc. Toutes ces informations ne seront pas utilisées dans le cadre de ce projet, mais peuvent avoir une importance lors de la comparaison entre les données de différents pays.

------------------------------------------------------------------------------

# 7. Prétraitement

------------------------------------------------------------------------------

Dues à la grande taille de la base de données, ces données brutes sont généralement de faible qualité. Elles peuvent être incomplètes (valeurs manquantes), bruitées (valeurs erronées ou aberrantes) ou incohérentes (divergence entre attributs). Il est donc nécessaire d'effectuer un prétraitement sur ces données, d’améliorer la qualité des données.

Dans cette partie, on va donc modifier, travailler les données à disposition grâce à `Pandas`, afin de supprimer les parties inutilisées, d'uniformiser les données, et ainsi ne travailler que sur des données pertinentes par la suite.

------------------------------------------------------------------------------

Étant donné que l'intégralité du projet se trouve dans ce notebook, il est vivement recommandé de ne jamais travailler directement sur le dataframe collecté, mais de passer par des copies pour pouvoir à tout moment récupérer à nouveau les données brutes.

------------------------------------------------------------------------------

## 7.1 Suppression des colonnes redondantes

------------------------------------------------------------------------------

Dans la base de données se trouvent des colonnes redondantes, colonnes n'apportant aucune nouvelle information telles que le nombre de nouveaux cas par millions d'habitants, le nombre de cas lissé, etc. Toutes ces colonnes ne seront pas utiles pour entrainer les modèles de machine learning, il faut donc les supprimer. 

------------------------------------------------------------------------------

## 7.2 Analyse des NaN (Not a Number)

------------------------------------------------------------------------------

Dans notre base de données se trouvent des colonnes qui sont peu remplies (case remplie avec NaN), en effet les données collectées sur ces sujets n'ont pas été récupérées tous les jours, ou n'ont pas été collectées depuis le début de l'épidémie.

En effet, on peut voir sur le graphique ci-dessous le taux de données non répertoriées (en beige) pour chacune des colonnes de la base de données. Certaines données ne sont pas récoltées chaque jour (ex: weekly hosp admissions), et d'autres n'ont commencé à être récoltées qu'un certain temps après le début de la pandémie (ex: new vaccinations).

------------------------------------------------------------------------------

Voici aussi, sous forme textuelle, le schéma présent ci-dessus:

------------------------------------------------------------------------------

Les données étant collectées que depuis moins de 2 ans, la base de données mise à disposition est donc de faible taille. Il a donc fallu garder un maximum de colonnes, sans pour autant fausser les résultats pouvant se baser sur une partie trop importante de données manquantes. Nous avons donc choisi de ne travailler que sur les colonnes contenant au moins 50% de données depuis le début de l'épidémie. Voici donc les différentes données restantes récoltées chaque jour disponibles après ce deuxième traitement:

------------------------------------------------------------------------------

## 7.3 Analyse de forme

------------------------------------------------------------------------------

Étant donné que le but du projet est de prédire le nombre de nouveaux morts par jour, nous allons travailler sur des modèles de machine learning basés sur la régression: prédire un nombre le plus proche de la réalité possible. Il est donc nécessaire de ne travailler que sur des données sous formes numériques, ainsi vérifions le type de données présentes.

> On doit tout de même garder la colonne `date` sous forme non numérique, car cette colonne deviendra par la suite l'axe des abscisses de tous les graphiques.

------------------------------------------------------------------------------

La colonne `tests_units` n'est pas sous forme numérique, il faut donc la supprimer de nos données. Pour ce faire, on supprime donc toutes les colonnes qui sont sous la forme 'object', cependant cette méthode supprime aussi la colonne `date`, il faudra donc la rajouter par la suite, mais pour l'instant la supprimer aussi n'a pas d'importance. Voici donc les différentes données restantes récoltées chaque jour disponibles après ce troisième traitement:

------------------------------------------------------------------------------

## 7.4 Analyse des corrélations

------------------------------------------------------------------------------

Afin de supprimer les colonnes ne permettant pas d'aider le modèle de machine learning à mieux prédire le nombre de morts chaque jour, il est nécessaire d'effectuer une analyse de corrélation entre les différentes informations contenues dans la base de données. La corrélation mesure une dépendance linéaire entre deux variables. L'analyse de corrélation permet donc d’étudier la dépendance entre le nombre de morts chaque jour et les autres informations à disposition.

On peut représenter ces corrélations grâce à une matrice de corrélation. Cette matrice de corrélation représente donc pour chaque information le niveau de dépendance avec les autres informations: lorsque la corrélation est proche de 1 (beige) ou -1 (noir) les deux informations sont dépendantes l'une de l'autre, plus cette valeur se rapproche de 0 moins la dépendance est forte.

------------------------------------------------------------------------------

Ici, il est seulement nécessaire de se focaliser sur les corrélations en rapport avec ce qui sera prédit: `new_deaths`. 

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

En observant attentivement le schéma du nombre de nouveaux morts chaque jour, il est possible d'apercevoir une saisonnalité, particularité que l'on ne retrouve que sur ce graphe. En effet, chaque week-end, le nombre de morts n'est pas forcément rempli, cela est flagrant entre octobre 2021 et janvier 2022: le nombre de morts chaque week-end est presque tout le temps nul. 

Cette saisonnalité est un réel problème dans cette base de données, car à conditions égales et en fonction du jour de la semaine, les données rapportées ne seront pas les mêmes. Le choix de supprimer l'intégralité des week-ends de la base de données a donc été fait, afin de ne pas induire le modèle de machine learning en erreur.

D'autres approches auraient pu être: de rajouter le jour dans une nouvelle colonne de la base de données, mais cette solution n'est pas suffisament prise en compte par le modèle de machine learning. Ou encore de remplacer le nombre de morts du week-end par la moyenne pondérée des jours précédents et suivants, or le but est de prédire des données, il est n'est donc pas viable de n'utiliser seulement les données des jours précédents.

TODO pas ouf comme explication

TODO schéma ?

------------------------------------------------------------------------------

On veut maintenant pouvoir remplir les données qui ne sont pas représentées. Pour ce faire, nous avons choisit de récupérer la valeur la plus proche pour chaque donnée manquante et remplir la base donnée avec ces informations. Cela permet de ne pas avoir de saut de données tout en gardant un maximum de données réelles pour entrainer le modèle de machine learning.

Une fois les données manquantes remplies, il faut supprimer les données négatives, car il est impossible qu'un nombre de morts soit négatif par exemple.

------------------------------------------------------------------------------

Le problème avec ce remplissage des données est que pour les colonnes `hosp_patients` et `icu_patients` ce remplissage n'est pas bien réalisé et ne peut pas être effectué simplement. Ces informations n'ont pas été récoltées depuis le début de la crise, et poseront problème au modèle. Pour palier à ce problème, nous avons choisi de supprimer l'intégralité des données récoltées avant le 30-03-2020.

------------------------------------------------------------------------------

Comme on peut le voir sur les graphes ci-dessus, une seule valeur est vraiment abérente compte tenu des données adjacentes, il est donc nécessaire de la supprimer. Cependant, les données sont bien trop disparates, et il n'est donc pas possible de supprimer cette donnée automatiquement sans pour autant modifier les informations jugées correctes. La valeur que l'on retrouve le 09-02-2020 est donc retirée manuellement, permettant ainsi de lisser les données.

------------------------------------------------------------------------------

A travers cette partie, le prétraitement et l'analyse des différentes données ont été réalisés. En effet, la base de données a été réduite afin de ne conserver que les données utiles permettant au mieux par la suite d'entrainer les modèles d'apprentissage automatique. Durant ce prétraitement, la base de données a été vidée de redondance de données, de valeurs Null ou négatives, de saisonnalité ou encore de données non corrélées au nombre de mort quotidien. La base de données est donc prête à servir pour l'entrainement de modèles d'apprentissage automatique.

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

Une possibilité lors de l'entrainement du modèle est l'`overfitting`: le modèle a tendance à essayer de recopier à la perfection les données d'entrainement, il ne sera donc plus capable de prédire des données inconnues car il est devenu trop spécialisé. Le cas contraire est aussi possible l'`underfitting`: le modèle généralise tellement les données qu'il a comme entrainement, qu'il n'est plus capable de prédire des valeurs précises. Ces deux problèmes surviennent lors du choix des paramères du modèle (partie qui sera traitée par la suite), il faut donc conserver un esprit critique car une évaluation parfaite face aux données d'entrainement n'implique pas forcément une prédiction parfaite par la suite.

------------------------------------------------------------------------------

## 8.2 Régression avec données

------------------------------------------------------------------------------

Maintenant que les données récoltées ont été traitées pour être utilisées au mieux, il va nous être possible de les utiliser afin d'entrainer un modèle de machine learning et de pouvoir prédire des données que nous ne connaissons pas encore. Dans notre cas, nous utiliserons l'apprentissage automatique afin de pouvoir prédire le nombre de nouveaux morts chaque jour à cause du Covid-19. 

Dans le cadre de ce projet, la temporalité des données est une informations très importante, nous allons donc travailler autour de cette caractéristique en posisitionnant la colonne `date` comme index du tableau de données.

------------------------------------------------------------------------------

**Division des données:**

Afin d'évaluer au mieux les performances d'un modèle de machine learning, il faut découper les données en trois grandes parties:
- les données d'entrainement (`train set`)
- les données de validation (`validation set`)
- les données de test (`test set`)

Pour commencer, il faut diviser en deux groupes les données de base: données de test, et données d'entrainement, en donnant un plus grande part aux données d'entrainement. Dans notre cas, il faut aussi prendre en compte la temporalité, il faut donc que les données d'entrainement soient au début, et les données de test à la fin. TODO 80%
Une fois les données divisées en deux catégories, il faut mettre de côté les données de test, ce sont les données qui nous permettront d'évaluer les différents modèles sur des données qu'ils ne connaissent pas.

<img src="https://i.imgur.com/JITVKcx.png" height="300">

Ensuite, pour évaluer un modèle lors de son entrainement, il est nécessaire de récupérer une partie des données restantes (données d'entrainement) afin de former les données de validation. Afin d'éviter tout biais dans l'apprentissage du modèle, il est nécessaire d'utiliser un `K-Fold cross validation` permettant de faire varier les données d'entrainement et les données de validation lors de l'apprentissage. Ce `K-Fold cross validation` divise en K parties les données d'entrainement, et forme une partie avec les données de validation, afin de tester toutes les combinaisons possibles d'entrainement.

<img src="https://i.imgur.com/VZrQkkD.png" height="300">

Cependant, dans ce projet, la temporalité des évènements importe énormément, il n'est donc pas viable d'utiliser un `K-Fold cross validation`, il faut plutot utiliser les `TimeSeriesSplit`. Cette classe permet d'effectuer de la même manière qu'un K-Fold, une séparation en K parties des données d'entrainement. Cependant, pour chaque entrainement du modèle, on utilise les données des entrainements précédents, ainsi que les 'nouvelles' données d'entrainement, le tout en fonction du temps.

<img src="https://i.imgur.com/ZDkrK1m.png" height="300">

**Evaluation des modèles:**

Après avoir entrainer les différents modèles de machine learning, il est nécessaire de les évaluer pour en déterminer les performances. Afin d'évaluer les différents modèles de machine learning, il faut déterminer une fonction d'évaluation qui sera utilisée par la suite pour évaluer tous les modèles. Pour ce faire, j'ai choisi la fonction `r2`, cette fonction représente le taux de corrélation des valeurs prédites avec les vraies données. Cette fonction d'évaluation a la particularité d'être normalisée, et d'être sous la forme d'un pourcentage, il est donc plus simple d'en interpréter les résultats. Cependant, il faut faire attention, car si les prédictions sont fortement anti-corrélées aux vraies données, le résultat sera aussi élevé.

**Comparaison avec une ligne de base:**

Pour voir si le modèle évalué est performant, il est nécessaire de le comparer avec une méthode dite 'naïve' de prédiction de données. En effet, si utiliser l'aléatoire est plus performant que de passer par le modèle de machine learning entrainé, c'est que le modèle n'est pas efficace. Comme dans ce projet, la temporalité des données est importante, j'ai choisi comme ligne de base de récupérer pour chaque jour le nombre de morts du jour précédent. Voici ci-dessous un exemple, avec les données représentés par des points orange, et la ligne de base décrite précédemment représentée par des traits orange. On voit ainsi bien le décalage d'un jour effectué.

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

La création d'un `Pipeline` permet d'appliquer des transformations sur les données et de spécifier le modèle à utiliser. En effet, certains algorithmes nécessitent une remise à l'échelle des données (données comprises entre 0 et 1 par exemple), pour ce faire il existe différentes classes permettant de le faire automatiquement telles que `MinMaxScaler`, ou `StandardScaler` par exemple. Ces classes permettant la remise à l'échelle des données seront définies et expliquées un peu plus loin lors de leur utilisation.

**Définition des paramètres du modèle:**

Pour chaque modèle, il existe un certains nombre de paramètres pouvant être modifiés afin d'augmenter les performances du modèle en question, ce sont les `hyper parameters`. Comme l'ajustement de ces paramètres est une tâche très complexe, et dépend du cas que l'on souhaite traiter, j'ai utilisé une grille référençant les différents paramètres à modifier, ainsi que leurs valeurs. Ainsi, toutes les combinaisons des paramètres seront testées afin d'en faire ressortir la meilleure par la méthode du `brut force`: test de toutes les combinaisons possibles.

**Utilisation d'un `GridSearch` pour combiner le tout:**

La classe `GridSearch` permet de combiner le `Pipeline` ainsi que la grille de paramètres, et d'autres fonctionnalités qui ne sont pas traitées dans le cadre de ce projet. Elle permet aussi de spécifier la fonction d'évaluation, la `cross validation`, et le nombre de coeurs de processeur utilisés. 

**Entrainement du modèle:**

Une fois la configuration terminée, il faut lancer l'entrainement des modèles sur les données d'entrainement.

**Evaluation du modèle:**

Afin d'évaluer les modèles, il est nécessaire de récolter le score de la fonction `r2` du modèle sur les données d'entrainement, les données de test, ainsi que sur la ligne de base pour pouvoir les comparer par la suite.

**Visualisation des résultats:**

Enfin, la visualisation du nombre de mort journalier permet de mieux représenter les résultats obtenus précédemment. En effet, pour chaque modèle, un graphe est construit représentant les données d'entrainement, les données de test, l'entrainement du modèle, la prédiction du modèle, ainsi que la ligne de base. Ces graphes sont interactifs: il est possible de ne retirer l'affichage de certaines données en cliquant sur la légende, de zoomer sur les axes en sélectionnant une zone, etc ... En survolant les données il est aussi possible de voir le détail de celles-ci. Enfin, pour une meilleure compréhension des graphiques réalisés, une légende y est associée: 
* `train`: le nombre de mort donné pour l'entrainement du modèle
* `test`: le nombre de mort à prédire
* `model fit`: l'approximation du modèle du nombre de mort durant son entrainement
* `prediction`: le nombre de mort que le modèle prédit 
* `baseline`: la référence reprenant le nombre de mort de la veille

> Sur ce graphe, seul le nombre de mort quotidien est représenté car c'est la donnée que l'on souhaite prédire, l'affichage des autres données n'apporterait aucune information supplémentaire mais ces données sont bien utilisées dans l'entrainement du modèle de machine learning.

------------------------------------------------------------------------------

### 8.2.1 Modèles linéaires

------------------------------------------------------------------------------

#### 8.2.1.1 Régression linéaire

------------------------------------------------------------------------------

### 8.2.2 K plus proches voisins (K Nearest Neighbors)

------------------------------------------------------------------------------

L'algorithme k-NN est un des algorithmes les plus simples à comprendre en machine learning. En effet, cet algorithme se base sur les k plus proches voisins afin de trouver la position du point à prédire, il réalise alors une moyenne (pondérée ou non) et détermine en utilisant les différents paramètres la valeur à prédire.

Les paramètres utilisés par cet algorithme sont: 
* `n_neighbors`: le nombre de voisins à utiliser
* `weights`: le poids de chacun des voisins (utilisation ou non de sa distance dans l'équation)
* `p`: la manière de calculer la distance (1=manhattan, 2=euclidienne [défaut])

Observations:
* L'utilisation de la distance des points dans le calcul du poids d'un voisin a une trop grande importance et le modèle `overfit` les données
* Il est important de ne pas donner un nombre k de voisins trop faible, sinon le modèle aura tendance à `overfit` les données.

------------------------------------------------------------------------------

### 8.2.3 Arbres de décision

------------------------------------------------------------------------------

#### 8.2.3.1 Arbre de décision simple

------------------------------------------------------------------------------

Les arbres de décisions simples sont des algorithmes permettant de représenter les données sous forme d'arbre et de règles. Plus la profondeur de l'arbre est grande, plus le nombre de règles augmente et plus l'arbre arrivera à recopier parfaitement la courbe. 

![](https://i.imgur.com/lQiGaRe.png)
> Exemple d'arbre de décision

Les paramètres utilisés par cet algorithme sont:
* `criterion`:
* `splitter`:
* `max_depth`: la profondeur maximale de l'arbre
 
Observations: 
* Il est important de ne pas donner un nombre trop élevé de profondeur, sinon le modèle aura tendance à `overfit` les données.

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

A Guide to the Types of Machine Learning Algorithms. https://www.sas.com/en_gb/insights/articles/analytics/machine-learning-algorithms.html. Accessed 25 Jan. 2022.

Brownlee, Jason. ‘How to Develop Multilayer Perceptron Models for Time Series Forecasting’. Machine Learning Mastery, 8 Nov. 2018, https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/.

---. Introduction to Time Series Forecasting With Python: How to Prepare Data and Develop Models to Predict the Future. Machine Learning Mastery, 2017.

COVID19 à Genève. Données Cantonales. https://infocovid.smc.unige.ch/. Accessed 25 Jan. 2022.

Covid-19: statistiques - République et canton de Neuchâtel. https://www.ne.ch/autorites/DFS/SCSP/medecin-cantonal/maladies-vaccinations/Pages/Covid-19-statistiques.aspx. Accessed 25 Jan. 2022.

‘Covid-19-Data/Public/Data at Master · Owid/Covid-19-Data’. GitHub, https://github.com/owid/covid-19-data. Accessed 25 Jan. 2022.

‘CovidTracker - Suivez l’épidémie de Covid19 en France et dans le monde’. CovidTracker, https://covidtracker.fr/. Accessed 25 Jan. 2022.

Data, S. R. F. ‘Coronavirus: les chiffres en Suisse’. SWI swissinfo.ch, https://www.swissinfo.ch/fre/donn%C3%A9es-actualis%C3%A9es_coronavirus--les-chiffres-en-suisse/45676324. Accessed 25 Jan. 2022.

‘Data Visualization With Seaborn and Pandas’. Hackers and Slackers, 29 Apr. 2019, https://hackersandslackers.com/plotting-data-seaborn-pandas/.

‘Decision Tree Regression’. Scikit-Learn, https://scikit-learn/stable/auto_examples/tree/plot_tree_regression.html. Accessed 25 Jan. 2022.

Google Colaboratory. https://colab.research.google.com/drive/1J8ZTI2UIJCwml2nrLVu8Gg0GXEz-7ZK0#scrollTo=vojrVZ9LEkgc. Accessed 25 Jan. 2022.

‘How to Automatically Install Required Packages From a Python Script?’ GeeksforGeeks, 3 Dec. 2021, https://www.geeksforgeeks.org/how-to-automatically-install-required-packages-from-a-python-script/.

‘Introduction to TensorFlow’. GeeksforGeeks, 4 Aug. 2017, https://www.geeksforgeeks.org/introduction-to-tensorflow/.

Jain, Mrinal. ‘Polynomial Regression with Keras’. Analytics Vidhya, 14 Mar. 2020, https://medium.com/analytics-vidhya/polynomial-regression-with-keras-ef1797b39b88.

Lengyel, Ivan. Pipreqsnb: Pipreqs with Jupyter Notebook Support. PyPI, https://github.com/ivanlen/pipreqsnb. Accessed 25 Jan. 2022.

Li, Susan. ‘An End-to-End Project on Time Series Analysis and Forecasting with Python’. Medium, 5 Sept. 2018, https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b.

Rajaa, Shangeth. ‘Polynomial Regression’. Shangeth, 29 Aug. 2019, https://shangeth.com/courses/deeplearning/1.2/.

Sher, Dr Varshita. ‘Time Series Modeling Using Scikit, Pandas, and Numpy’. Medium, 24 Mar. 2021, https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1.

Shrivastava, Soumya. ‘Cross Validation in Time Series’. Medium, 17 Jan. 2020, https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4.

‘Support Vector Regression In Machine Learning’. Analytics Vidhya, 27 Mar. 2020, https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/.

Support Vector Regression (SVR) Using Linear and Non-Linear Kernels — Scikit-Learn 0.18.2 Documentation. https://scikit-learn.org/0.18/auto_examples/svm/plot_svm_regression.html. Accessed 25 Jan. 2022.

Tableau de Bord COVID-19. https://www.gouvernement.fr/info-coronavirus/carte-et-donnees. Accessed 25 Jan. 2022.

Tous les modèles de Machine Learning expliqués en 8 minutes. 16 Jan. 2020, https://moncoachdata.com/blog/modeles-de-machine-learning-expliques/.

‘Which Machine Learning Algorithm Should I Use?’ The SAS Data Science Blog, https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/. Accessed 25 Jan. 2022.


------------------------------------------------------------------------------

## 10.2 Sources illustratives

------------------------------------------------------------------------------

Image KFold: https://www.researchgate.net/figure/The-technique-of-KFold-cross-validation-illustrated-here-for-the-case-K-4-involves_fig10_278826818

------------------------------------------------------------------------------

