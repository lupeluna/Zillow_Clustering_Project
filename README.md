Hello,

Welcome to the README file for my Zillow Clustering Project.

Here, you will find expanded information on this project including goals, how I will be working through the pipeline and a data dictionary to help offer more insight to the variables that are being used.

## Goal
 * Identify the drivers for errors in Zestimates by incorporating clustering methodologies.
 * Document the process and analysis throughout the data science pipeline.
 * Demonstrate our process and articulate the information that was discovered.
Deliverables:
 - README.md file containing overall project information, how to reproduce work, and notes from project planning.
 - Jupyter Notebook Report detailing the pipeline process.
 - Python modules that automate the data acquistion, preparation, and exploration process.

______________________



## Planning process

### Project Outline:

For this project we are working with the Zillow dataset using the 2017 properties and predictions data for single unit / single family homes.

This notebook consists of discoveries made and work that was done related to uncovering what the drivers of the error in the zestimate is.

 * Acquisiton of data through Codeup SQL Server, using env.py file with username, password, and host
 * Prepare and clean data with python - Jupyter Labs / Jupyter Notebook
 * Explore data
    * if value are what the dictionary says they are
    * null values
      * are the fixable or should they just be deleted
    * categorical or continuous values
    * Make graphs that show relationships
 * Use clustering to create a subsection of the data
 * Run statistical analysis
 * Model data using regression algorithms
 * Test Data
 * Conclude results




###  Data Dictionary
- The dictionary below is a reference for the variables used within the dataset



|   Feature      |  Data Type   | Description    |
| :------------- | :----------: | -----------: |
|  parcelid | int64   | Unique parcel identifier    |
| bedroomcnt    | float64 | count of bedrooms |
| bathroomcnt   | float64 | count of bathrooms |
|  fips  | float64   | county code - see below    |
| county  # 6037| float64 | Los Angeles |
| county # 6059 | float64   | Orange    |
| county # 6111  | float64| Ventura |
| age    | float64 | age of home|
| tax_rate    | float64 | This is property tax / tax_assessed_value|



-------------------
 
 
#### Initial Hypotheses

> - **Hypothesis 1 -** I rejected the Null Hypothesis; there is a difference.
> - alpha = .05
> - $H_o$: There is no association between number of bedrooms and logerror.  
> - $H_a$: There is an association between number of bedrooms and logerror. 

> - **Hypothesis 2 -** I rejected the Null Hypothesis; there is a difference.
> - alpha = .05
> - $H_o$: There is no association between structure_dollar_per_sqft and logerror.
> - $H_a$: There is an association between structure_dollar_per_sqft and logerror.


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Recommendations & next steps:

 * The goal of identifying the drivers for errors in Zestimates by incorporating clustering methodologies helped, but not by much.

 * With more time:

   * would like to find if there are better predictors of log error.
   * would like to explore 3-D clustering by adding latitude or acres.
   * explore less common features like a/c unit type and fireplaces.
   * would like to fill out the missing data so that there are even more data points to work with.



<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### **Plan ->** Acquire -> Prepare -> Explore -> Model -> Deliver
- [x] Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- [x] Acquire data from the SQL Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- [x] Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- [x]  Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- [x] Establish a baseline accuracy and document well.
- [x] Train three different regression models.
- [x] Evaluate models on train and validate datasets.
- [x] Choose the model with that performs the best and evaluate that single model on the test dataset.
- [x] Create csv file with the customer id, the probability of churn, and the model's predictions.
- [x] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

#### Acquire
> - Store functions that are needed to acquire data  that will be used for the Zillow Regression Project
> - The final function will return a pandas DataFrame


#### Prepare
> - Store functions needed to prepare the Zillow data
> - Import the prepare functions created by using .prepare.py


#### Explore
> - Answer key questions, my hypotheseses, and figure out the features that can be used in a regression model to best predict driver for churn
> - Cluster the data

#### Model
> - Establish a baseline accuracy to determine if having a model is better than no model and train for at least 3 different models

#### Deliver
> - Deliver my findings in the presention.



<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [X] Read this README.md
- [ ] Download the aquire.py, prepare.py, and final_report.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook