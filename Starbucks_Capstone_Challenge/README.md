# Project Description: Mining Starbucks customer data - predicting offer success

**[BLOGPOST](https://gonzalo-munillag.medium.com/starbucks-challenge-accepted-ded225a0867)**

## Table of Contents
1. [Introduction and motivation](#Introduction_and_motivation)
2. [Installation](#Installation)
3. [Files in the repository](#files)
4. [Results](#Results)
5. [Details](#Details)
6. [Data Sets](#Data)

### Introduction and motivation <a name="Introduction_and_motivation"></a>

This project aims to answer a set of questions based on the provided datasets from Starbucks: transactions, customer profiles and offer types. 
The main question we will ask, and around which the whole project revolves, is:

                What is the likelihood that a customer will respond to a certain offer?

Other questions to be answered are:

About the offers:
- Which one is the longest offer duration?
- Which one is the most rewarding offer?

About the customers:
- What is the gender distribution?
- How different genders are distributed with respect to income?
- How different genders are distributed with respect to age?
- What is the distribution of new memberships along time?

About the transactions:

- Which offers are preferred according to gender?
- Which offers are preferred according to income?
- Which offers are preferred according to age?
- Which offers are preferred according to date of becoming a member?
- Which are the most successful offers?
- Which are the most profitable offers?
- Which are the most profitable offers between informational?
- How much money was earned in total with offers Vs. without offers?

**The motivation is to improve targeting of offers to Starbucks' customers to increase revenue.**

**Process and results presented in this [blogpost](https://gonzalo-munillag.medium.com/starbucks-challenge-accepted-ded225a0867).**

We will follow the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) data science process standard for accomplishing the data analysis at hand.

### Installation <a name="Introduction_and_motivation"></a>

**Packages needed**
1. Wrangling and cleansing: pandas, json, pickle
2. Math: numpy, math, scipy
3. Visualization: matplotlib, IPython, 
4. Progress bar: tim, progressbar
5. ML: sklearn

### Files in the repository <a name="files"></a>

1. data folder:
    1.1 portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
    1.2 profile.json - demographic data for each customer
    1.3 transcript.json - records for transactions, offers received, offers viewed, and offers completed
2. Starbucks_Capstone_notebook.ipynb: Contains an executable python notebook for your to execute and modify as you wish.
3. Starbucks_Capstone_notebook.html: If you are not interested in extending or executing the code yourself, you may open this file and read through the anaylsis.
4. Other pickle files saving the datasets and models.

### Results <a name="Results"></a>

The best model to predict if an offer will be successful is Gradient Boosting.
However, 70% is not such a high accuracy, better than human though. 
Grid search did not show much improvements, so furtehr tunning should be carried out.
We saw that the learning rate went from 0.1 to 0.5, while the rest of parameters stayed the same. The enxt logical step would be to try with a learning rate of 0.75 (as 1 was not chosen) and try to change other parameters.


### Details <a name="Details"></a>

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

#### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

#### Cleaning

This makes data cleaning especially important and tricky.

You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.

####  Advice

Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

### Data Sets <a name="Data"></a>

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

The data is contained in three files:

    portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
    profile.json - demographic data for each customer
    transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:
