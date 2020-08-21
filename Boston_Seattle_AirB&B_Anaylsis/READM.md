## Project motivation

The project aims to analyse part of the data from AirB&B of the cities of Boston and Seattle by answering 4 questions:

1. How has price developed over time? How much bigger in average have been the prices of Boston in contrast to Seattle throught 2016 and 2017? What is the difference between the all time high and low in avergae price in Boston and in Seattle and what is the ratio between the deltas?
2. What are the attributes that best explain the price of the households advertized and is it possible to predict the price with the given data?
3. What are the features that best explain the listings dataset for Boston and how are these correlated?
4. What are the attributes that best explain good review rating scores of the households advertized and is it possible to predict the reviews with the given data?

[Blogpost](https://munigarry.wixsite.com/bostonseattle)

## Libraries used

Numpy, pandas, matplotlib, csv, random, datetime, seaborn, sklearn.

## File in the repository:

1. Boston_Seattle_AirBNB_co_analysis.ipynb: Contains an executable python notebook for your to execute and modify as you wish.
2. Boston_Seattle_AirBNB_co_analysis.html: If oyu are not interested in extending or executingthe code yourself, you may open this file and read through the anaylsis.

## Results

The first conclusion is that the prices of AirB&Bs in Boston (red) are much greater overtime in average, a 45%. The second conclusion is that the prices for Boston fluctuate more than in Seattle, the deltas are more erratic. The third conclusion is that the prices of Boston have a downward trend, while in Seattle have an upward trend. And the fourth conclusion is that the maximum delta of prices in Boston is 176% higher than in Seattle. 
The downward trend on Boston at the begining might be due to a hype of AirB&B when it was first released in Boston. Then market forces reduced the avergae prices of the households and around halfway on its trendline, the prices start to pick up again in a mild upward trend, like the one Seattle had in the beginning. 
Another interesting feature of the Boston curve to check is the spike that sprouts around April 2017. Doing some googling, I found the event that might have caused the spike, a massive marathon (26k runners): https://en.wikipedia.org/wiki/2017_Boston_Marathon

Thus we can conclude that the attributes that best explain price are the accomodation itself, the number of people that can be accomodated and the size of the apartment (square fit, and number of rooms).

The analysis for prices and reviews of Seattle were similar to the ones from Boston.

Something to note is that the price and review predcition was not accurate for these datasets, we conclude that we need more data.

