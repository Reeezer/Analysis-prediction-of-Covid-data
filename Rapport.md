# P3 - Analyse et prédiction de données Covid-19

------------------------------------------------------------------------------

# Introduction

------------------------------------------------------------------------------

Depuis le début de la crise sanitaire de Covid-19 en 2020, nous avons connu un grand changement dans nos vies quotidiennes. Les routines ont changé, ainsi que notre de travailler, ou encore d'interagir et même de nous comporter. Chaque jour, un grand nombre de données relatives au Covid-19 sont collectées sur le nombre de patients infectés, testés, décédés, guéris ou vaccinés. Les données sur les différentes mutations et les différents vaccins dans tous les pays du monde sont soigneusement collectées et pas toujours analysées ou visualisées de manière à pouvoir tirer certaines conclusions.

TODO agrandir l'introduction

Dans ce projet, l'objectif est donc de collecter et analyser les données Covid-19 pour différents pays et régions. Nous ferons des analyses statistiques et proposerons différentes méthodes de visualisation. En outre, nous essaierons de faire des prédictions en utilisant certains des outils et modules fournis par Python.

Ce projet se découpe donc en trois grandes parties : récupération de données, visualisation des données récupérées, ainsi que prédictions des futurs évènements.

------------------------------------------------------------------------------

## Récupération de données

Au début du projet, il faudra chercher et choisir une bases de données viable et libre d'accès pour entrainer le modèle de machine learning. Il faudra ensuite choisir les données à traiter, ainsi que mettre en place un manière de traiter les données récoltées.

------------------------------------------------------------------------------

## Visualisation des données

Par la suite, il faudra chercher, choisir et représenter les données sous la forme la plus adaptée et compréhensible possible (graphiques, diagrammes, ...). Par la suite, il sera nécessaire de choisir les différents paramètres pris en compte pour l'affichage des données (catégorie, temporalité, ...). 
Pour cela, il faudra au préalable prendre en main des outils de visualisation tels que Pandas ou Seaborn.

------------------------------------------------------------------------------

## Analyse de données et prédictions

Enfin, il faudra chercher, comparer, choisir et tester le/les modèle(s) de machine learning utilisé(s) permettant la prédiction du nombre de nouveaux morts chaque jour. Pour cela, il faudra au préalable prendre en main des outils de machine learning tels que Scikit-learn. Pour finir, le nombre de nouveaux morts chaque jour devra être affiché en rapport aux données récoltées au préalable.

------------------------------------------------------------------------------

## Bibliothèques utilisées

------------------------------------------------------------------------------

Visualisation: 
- Matplotlib: Diagrammes simples et compréhensifs
- Seaborn: Diagrammes complex et complets
- Plotly: Diagrammes interactifs

Représentation:
- Numpy: Représentation simpliste des données
- Pandas: Représentation des données sous forme de tableau labélisé

Machine learning:
- Scikit-learn: Modèles de machine learning, évaluation des modèles, mise en forme des données

------------------------------------------------------------------------------

# Etat de l'art

------------------------------------------------------------------------------

TODO

Etat de l'art

Domaines dans lesquels on pourrait utiliser aussi ce projet

------------------------------------------------------------------------------

# Récupération des données

------------------------------------------------------------------------------

Our World In Data est un site internet recensant un nombre impressionnant de données sur tout type de sujets à travers le monde. Les publications présentent sur le site sont dirigées par l'université d'Oxford et rédigées par l'historien social et économiste du développement Max Roser.

Ce site internet met à disposition une base de données accessible via GitHub: https://github.com/owid/covid-19-data/tree/master/public/data, ou directement sur leur site internet: https://ourworldindata.org/coronavirus.

Il y a à disposition plusieurs formats afin de récupérer cette base de données: csv, xlsx ou json. Json étant un format plus simple de compréhension, c'est celui que j'ai choisi de traiter. Cette base de données est tenue à jour et est actualisée chaque jour, ainsi les données traitées dans ce projet seront toujours actualisées.

Grâce à la bibliothèque Pandas, on peut récupérer les données en passant l'url de recherche à la méthode `read_json`. 

------------------------------------------------------------------------------

Une fois les données récoltées, il va falloir les traiter afin d'en extraire les parties importantes et utiles. 

Dans le cadre de ce projet, nous allons traiter les données relatives à la Suisse, en utilisant donc l'ISO `CHE`. Les données Covid de chaque jour, se trouvent ensuite dans la partie `data` de la base de données. 

On peut voir avec `tail`, que les données sont bien actualisées chaque jour. On peut aussi voir la forme des données, ainsi que les différentes informations contenues dans cette base de données. En effet, 45 données sont récoltées chaque jour en Suisse, ce nombre peut être différent en fonction du pays que l'on souhaite traiter.

------------------------------------------------------------------------------

Dans cette base de données, se trouvent aussi quelques informations à propos de chaque pays: le continent, la population, l'age médian, le nombre de lits d'hopitaux ... Toutes ces informations ne seront pas utilisées dans le cadre de ce projet, mais peuvent avoir une importance lors de la comparaison entre les données de différents pays.

------------------------------------------------------------------------------

# Prétraitement

------------------------------------------------------------------------------

Dû à la grande taille de la base de données, ces données brutes sont généralement de faible qualité. Elles peuvent être incomplètes (valeurs manquantes), bruitées (valeurs erronées ou aberrantes) ou incohérentes (divergence entre attributs). Il est donc nécessaire d'effectuer un prétraitement sur ces données, d’améliorer la qualité des données.

Dans cette partie, on va donc modifier, travailler les données à disposition grâce à Pandas, afin de supprimer les parties inutilisées, d'uniformiser les données ... 

------------------------------------------------------------------------------

Etant donné que l'intégralité du projet se trouve dans ce notebook, il est vivement recommandé de ne jamais travaillé directement sur le dataframe collecté, mais de passer par des copies pour pouvoir à tout moment récupérer les données brutes.

------------------------------------------------------------------------------

## Suppression des colonnes redondantes

------------------------------------------------------------------------------

Dans la base de données se trouvent des colonnes redondantes, colonnes n'apportant aucune nouvelle informations telles que le nombre de nouveaux cas par millions d'habitants, le nombre de cas lissé, ... Toutes ces colonnes ne seront pas utiles pour entraîner les modèles de machine learning, il faut donc les supprimer. Voici donc les différentes données restantes récoltées chaque jour disponibles après ce premier traitement:

------------------------------------------------------------------------------

## Analyse des NaN

------------------------------------------------------------------------------

Dans notre base de données, se trouve des colonnes qui sont peu remplies (case remplie avec NaN), en effet les données collectées sur ces sujets n'ont pas été récupérées tous les jours, ou n'ont pas été collectées depuis le début de l'épidémie.

En effet, on peut voir sur le graphique ci-dessous le taux de données non répertoriées (en beige) pour chacune des colonnes de la base de données. Certaines données ne sont pas récoltées chaque jour (ex: weekly hosp admissions), et d'autres n'ont commencé à être récoltées qu'un certains temps après le début de la pandémie (ex: new vaccinations).

------------------------------------------------------------------------------

Les données étant collectées que depuis moins de 2 ans, la base de données mise à disposition est donc de faible taille. J'ai donc choisi de ne travailler que sur les colonnes contenant au moins 50% de données depuis le début de l'épidémie.

------------------------------------------------------------------------------

## Analyse de forme

------------------------------------------------------------------------------

Etant donné que le but du projet est de prédire le nombre de nouveaux morts par jour, nous allons travailler sur des modèles de machine learning basés sur la régression: chercher le nombre le plus proche de la réalité. Il est donc nécessaire de travailler sur des données sous formes numériques, ainsi vérifions le type de données présentes.

On garde cependant `date` sous forme non-numérique car cette colonne deviendra par la suite l'indice du tableau.

------------------------------------------------------------------------------

La colonne `tests_units` n'est pas sous forme numérique, il faut donc la supprimer de nos données. Pour ce faire on supprime donc toutes les colonnes qui sont sous la forme object, cependant cette méthode supprime aussi la colonne `date`, il faudra donc la rajouter par la suite, mais pour l'instant la supprimer aussi n'a pas d'importance.

------------------------------------------------------------------------------

## Analyse des corrélations

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

## Mise en forme

------------------------------------------------------------------------------

Une fois les colonnes utiles sélectionnées, il est nécessaire d'effectuer une mise en forme des données disponibles car un certains nombre de données ne sont pas représentées, ou sont négatives or c'est tout simplement impossible.

On peut donc voir ci-après les différentes informations en fonction du temps:

------------------------------------------------------------------------------

On veut donc pouvoir remplir les données qui ne sont pas représentées. Pour ce faire, j'ai choisi de récupérer la valeur la plus proches pour chaque donnée manquante et remplir la base données avec ces informations. Cela permet de ne pas avoir de saut de données tout en gardant un maximum de données réelles pour entrainer le modèle de machine learning.

Une fois les données manquantes remplies, il faut supprimer les donnnées négatives car il est impossible qu'un nombre de morts soit négatif par exemple.

------------------------------------------------------------------------------

Le problème avec ce remplissage des données est que pour les colonnes `hosp_patients` et `icu_patients` ce remplissage n'est pas bien réalisé et ne peut pas être effectué simplement. Pour éviter de supprimer trop de données, je vais donc supprimer l'intégralité des données récoltées avant le 30-03-2020.

------------------------------------------------------------------------------

On veut enfin supprimer la valeur exrême qu'on retrouve le 09-02-2020, cependant comme le nombre de morts est très disparate à travers le temps, j'ai choisi de la retirer à la main de la base de données. 

------------------------------------------------------------------------------

# Prédictions

------------------------------------------------------------------------------

## Définitions

------------------------------------------------------------------------------

Ces dernières années, les méthodes d'apprentissage automatique sont devenues omniprésentes dans la vie quotidienne. Des recommandations automatiques de films à regarder, de plats à commander ou de produits à acheter, la radio en ligne personnalisée ou la reconnaissance de vos amis sur vos photos, de nombreux sites Web et appareils modernes sont dotés d'algorithmes d'apprentissage automatique.

Mais tout d'abord, qu'est-ce que le l'apprentissage automatique ? L'apprentissage automatique, aussi appelé "machine learning", est une sous-catégorie de l'intelligence artificielle qui se résume en la capacité d'une machine à imité le comportement humain. L'intelligence artificielle est utilisée pour effectuer des tâches complexes d'une manière similaire à la façon dont les humains résolvent les problèmes.

------------------------------------------------------------------------------

### Types d'algorithmes

------------------------------------------------------------------------------

Il existe deux types d'algorithmes d'apprentissage:
- Supervisé: algorithmes qui automatisent le processus de décision en généralisant à partir d'exemples connus. Dans ce cas, l'utilisateur fournit à l'algorithme des paires d'entrées et de sorties souhaitées, et l'algorithme trouve un moyen de produire la sortie souhaitée à partir d'une entrée. En effet, l'algorithme est capable de créer une sortie pour une entrée qu'il n'a jamais vue auparavant sans aucune aide.
- Non supervisés: dans ce type d'algorithmes, seules les données d'entrée sont connues, et aucune donnée de sortie connue n'est donnée à l'algorithme. Bien qu'il existe de nombreuses applications réussies de ces méthodes, elles sont généralement plus difficiles à comprendre et à évaluer.

Dans notre cas, comme les données à prédire sont connues, nous utiliserons des algorithmes d'apprentissage supervisé.

------------------------------------------------------------------------------

### Types de problèmes

------------------------------------------------------------------------------

Dans ce type d'algorithmes, il existe deux types de problèmes d'apprentissage supervisé:
- Regression: l'objectif est de prédire un nombre continu. (Ex: prédire le prix d'une maison)
- Classification: l'objectif est de prédire une classe, qui est un choix parmi une liste prédéfinie de possibilités. (Ex: prédire la race d'un animal)

Dans notre cas, comme les données à prédire sont le nombre de nouveaux morts par jour, nous sommes dans un cas de régression.

------------------------------------------------------------------------------

### Caractéristique d'un modèle

------------------------------------------------------------------------------

Dans l'apprentissage supervisé, nous voulons construire un modèle à partir de données, et être ensuite capable de faire des prédictions précises sur de nouvelles données. Si un modèle est capable de faire des prédictions précises sur des données inconnues, on dit qu'il est capable de généraliser de l'ensemble d'apprentissage à l'ensemble de test. Nous voulons construire un modèle capable de généraliser aussi précisément que possible.

------------------------------------------------------------------------------

## Première approche: Régression simple en utilisant seulement la date

------------------------------------------------------------------------------

Maintenant que les données récoltées ont été traitées pour être utilisées au mieux, il va nous être possible de les utiliser afin d'entrainer un modèle de machine learning et de pouvoir prédire des données que nous ne connaissons pas encore. 
Dans notre cas, nous utiliserons l'apprentissage automatique afin de pouvoir prédire le nombre de nouveaux morts chaque jour à cause du Covid-19. 

------------------------------------------------------------------------------

## Linear Models


------------------------------------------------------------------------------

### Ordinary Least Squares

------------------------------------------------------------------------------

### Ridge regression

------------------------------------------------------------------------------

### Lasso

------------------------------------------------------------------------------

### Logistic Regression

------------------------------------------------------------------------------

## Naive Bayes Classifiers

------------------------------------------------------------------------------

### Gaussian

------------------------------------------------------------------------------

### Multinomial

------------------------------------------------------------------------------

## K-Nearest Neighbors

------------------------------------------------------------------------------

## Decision Trees

------------------------------------------------------------------------------

### Simple Decision Trees

------------------------------------------------------------------------------

### Random Forest

------------------------------------------------------------------------------

### Gradient Boosted Trees

------------------------------------------------------------------------------

## Support Vector Regression

------------------------------------------------------------------------------

## Multi-Layer Perceptrons (Deep Learning)

------------------------------------------------------------------------------

# Testing machine learning models with multiple features to predict every day new deaths

------------------------------------------------------------------------------

Décaler les y vers la gauche pour pouvoir prédire sur le mois d'après

Knn pour prédire la forme des features, et ensuite utiliser ces features approximée pour approximer la feature à prédire

------------------------------------------------------------------------------

## K-Nearest Neighbors

------------------------------------------------------------------------------

## Decision Trees

------------------------------------------------------------------------------

### Simple Decision Trees

------------------------------------------------------------------------------

### Random Forest

------------------------------------------------------------------------------

### Gradient Boosted Trees

------------------------------------------------------------------------------

## Support Vector Regression

------------------------------------------------------------------------------

## Multi-Layer Perceptrons (Deep Learning)

------------------------------------------------------------------------------

# Décalage des features pour faire de la prédiction

------------------------------------------------------------------------------

