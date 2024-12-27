<div id="top"></div>

<!-- PROJECT NAME -->
<br />
<div align="center">

  <h1 align="center">Stiker Recruitment</h1>

  <h3 align="center">
    Assessment of potential striker targets with recommendations
  </h3>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About this project

This project performs analysis on events and matches data to assess stiker targets against the requirements of the club. The analysis uses the events and matches data to calculate each target's performance across a variety of key metrics, focusing mainly on receptions in the danger zone and goal scoring abilities. To contextualise player metrics for opponent level and match competitiveness, key player metrics have additionally been normalised for opponent ELO rating and difference in ELO rating of opponent from player's team. To compare performance of targets across key metrics, both with and without normalisation, box plots, scatter plots and radar plots are produced, with outputs saved for use in report writing.
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting started

### Prerequisites

The following are prerequisites to run this codebase:
 - Python
 - Poetry

 ### Installation

1. Install the poetry environment by running the following in the terminal

   ```sh
   poetry install
   ```

2. Save the following CSV files into the data/raw folder

    * EventLevelData.csv
    * MatchLevelDataCorrectedMapping.csv
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->

## Usage
There are two main Python files that need to be run to generate the outputs of the project.

1. Firstly, the <b>'data_munging.py'</b> file needs to be run. This file reads in the raw events and matches data, transforms them and merges them to create a single input dataset to the analysis. The main transformation steps involve labelling events within the same player possession and defining normalisation weightings for matches based off opponent ELO ratings and the difference in opponent ELO rating from a player's team ELO rating. After running this Python file two CSV files will be saved in the data/processed folder. A transformed version of the events data, 'TransformedEventLevelData.csv', and the merged input dataset for the analysis, 'MergedEventsMatchLevelData.csv'.

2. Secondly, the <b>'data_analysis.py'</b> file needs to be run. This file reads in the 'MergedEventsMatchLevelData.csv' dataset, calculates key player performance metrics and then creates visualisations to compare performance metrics across players. To calculate the desired metrics the input event-level dataset is first transformed to possession-level, where each row represents a single player possession. For each possession fields providing information around the match, team, player, start event and end event are included, along with the normalisation weightings calcluated in step 1. This possessions dataset is used to calcluate multiple player level metrics. A focus is placed on metrics that assess a player's ability to receive the ball in dangerous areas and create goal scoring opportunities, as well as the player's overall goal scoring abilities. In addition to the pure player-level metrics, the normalisation weightings for opponent ELO rating and ELO rating difference of opponent from player's team are utilised to generate normalised player-level metrics. Metrics are normalised at a match and player level by multiplying pure metric values by opposition ELO weighting and ELO difference weighting. Metrics are then aggregated to player-level through a weighted average based on minutes played. The aggregated normalised metrics are finally each scaled using a 0-100 scale with 100 representing the maximum value of the metric across all players. To compare metrics, both before and after normalisation, a range of plots are produced and saved in the data/outputs folder, including:
    - Box plots for opponent ELO rating and difference in opponent ELO rating from player's team by player
    - Pitch plots for the proportion of receptions in each of the 18 pitch zones for each player
    - Scatter plots of shots and shot assists per 90 from receptions in danger zone against against xG per 90 from receptions in danger zone by player, for both pure and normalised values
    - Scatter plots of xG per 90 against goals per 90 by player, for both pure and normalised values
    - Radar plot of key normalised (or just scaled if unable to be normalised) metrics by player

<p align="right">(<a href="#top">back to top</a>)</p>